import random
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict

from mmdet import datasets
from mmdet.core import (CocoDistEvalmAPHook, CocoDistEvalRecallHook,
                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.models import RPN
from mmdet.utils import get_root_logger


from mmdet.datasets import build_dataset
import mmcv
import pycocotools.mask as mask_util
from mmdet.core import coco_eval, results2json, results2json_segm, wrap_fp16_model, tensor2imgs, get_classes
from .comm import synchronize

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model,
                dataset,
                cfg,
                validate=False,
                logger=None,
                timestamp=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # # if model.module.bbox_head.freeze_solov2_and_train_combonly:
    # if model.module.bbox_head.optimize_list is not None:
    #     for (key, param) in model.named_parameters():
    #         # if 'kernel_convs_convcomb' not in key and 'context_fusion_convs' not in key and 'learned_weight' not in key:
    #         if not any(s in key for s in model.module.bbox_head.optimize_list):
    #             param.requires_grad=False
    #         else:
    #             # print('optimize {}'.format(key))
    #             logger.info('optimize {}'.format(key))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, logger=logger)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = DATASETS.get(val_dataset_cfg.type)
            if issubclass(dataset_type, datasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    ## add test after training
    if cfg.data.test.ann_file != 'data/lvis/lvis_v0.5_val_lvis_freqset.json': # if val set is lvis freq, only eval on lvis-freq val set
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        model_orig=model.module
        model = MMDataParallel(model, device_ids=[0]).cuda()
        # data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)

        ##eval on lvis-77######
        cfg.data.test.ann_file = 'data/lvis/lvis_v0.5_val_cocofied.json'
        cfg.data.test.img_prefix = 'data/lvis/val2017/'
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # model_orig=model.module
        # model = MMDataParallel(model, device_ids=[0]).cuda()
        # data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    from lvis import LVISEval
                    lvisEval = LVISEval('data/lvis/lvis_v0.5_val_cocofied.json', result_files, 'segm')
                    lvisEval.run()
                    lvisEval.print_results()
                    # fix lvis api eval iou_thr error, should be 0.9 but was 0.8999
                    lvisEval.params.iou_thrs[8] = 0.9
                    for iou in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        print('AP at iou {}: {}'.format(iou, lvisEval._summarize('ap', iou_thr=iou)))
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)
    else:
        ##eval on lvis-freq######
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # model_orig=model.module
        # model = MMDataParallel(model, device_ids=[0]).cuda()
        data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    from lvis import LVISEval
                    lvisEval = LVISEval(cfg.data.test.ann_file, result_files, 'segm')
                    lvisEval.run()
                    lvisEval.print_results()
                    # fix lvis api eval iou_thr error, should be 0.9 but was 0.8999
                    lvisEval.params.iou_thrs[8] = 0.9
                    for iou in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        print('AP at iou {}: {}'.format(iou, lvisEval._summarize('ap', iou_thr=iou)))
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)

def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int)
        cate_score = cur_result[2].cpu().numpy().astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)

        return masks

def single_gpu_test(model, data_loader, show=False, verbose=True):
    model.eval()
    results = []
    dataset = data_loader.dataset

    num_classes = len(dataset.CLASSES)
    if dataset.ann_file=='data/lvis/lvis_v0.5_val_lvis_freqset.json':
        print('warining: eval with lvis-freq, change num_classes to 513')
        num_classes=513
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=not show, **data)

        result = get_masks(seg_result, num_classes=num_classes)
        results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None):
    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # # if model.module.bbox_head.freeze_solov2_and_train_combonly:
    # if model.module.bbox_head.optimize_list is not None:
    #     for (key, param) in model.named_parameters():
    #         # if 'kernel_convs_convcomb' not in key and 'context_fusion_convs' not in key and 'learned_weight' not in key:
    #         if not any(s in key for s in model.module.bbox_head.optimize_list):
    #             param.requires_grad=False
    #         else:
    #             # print('optimize {}'.format(key))
    #             logger.info('optimize {}'.format(key))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, logger=logger)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    ## add test after training
    if cfg.data.test.ann_file != 'data/lvis/lvis_v0.5_val_lvis_freqset.json': # if val set is lvis freq, only eval on lvis-freq val set
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        model_orig=model.module
        model = MMDataParallel(model, device_ids=[0]).cuda()
        data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)

        ##eval on lvis-77######
        cfg.data.test.ann_file = 'data/lvis/lvis_v0.5_val_cocofied.json'
        cfg.data.test.img_prefix = 'data/lvis/val2017/'
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # model_orig=model.module
        # model = MMDataParallel(model, device_ids=[0]).cuda()
        data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    from lvis import LVISEval
                    lvisEval = LVISEval('data/lvis/lvis_v0.5_val_cocofied.json', result_files, 'segm')
                    lvisEval.run()
                    lvisEval.print_results()
                    # fix lvis api eval iou_thr error, should be 0.9 but was 0.8999
                    lvisEval.params.iou_thrs[8] = 0.9
                    for iou in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        print('AP at iou {}: {}'.format(iou, lvisEval._summarize('ap', iou_thr=iou)))
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)
    else:
        ##eval on lvis-freq######
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # model_orig=model.module
        # model = MMDataParallel(model, device_ids=[0]).cuda()
        data_loader.dataset.img_infos = data_loader.dataset.img_infos[:100]
        outputs = single_gpu_test(model, data_loader)

        print('\nwriting results to {}'.format('xxx'))
        # mmcv.dump(outputs, 'xxx')
        eval_types = ['segm']
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = 'xxx'
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json_segm(dataset, outputs, 'xxx', dump=False)
                    from lvis import LVISEval
                    lvisEval = LVISEval(cfg.data.test.ann_file, result_files, 'segm')
                    lvisEval.run()
                    lvisEval.print_results()
                    # fix lvis api eval iou_thr error, should be 0.9 but was 0.8999
                    lvisEval.params.iou_thrs[8] = 0.9
                    for iou in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        print('AP at iou {}: {}'.format(iou, lvisEval._summarize('ap', iou_thr=iou)))
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = 'xxx' + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file, dump=False)
                        coco_eval(result_files, eval_types, dataset.coco)