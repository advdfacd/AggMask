import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, roi_align
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule

INF = 1e8

from scipy import ndimage


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d


@HEADS.register_module
class AggMaskStarHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_kernel_for_comb,
                 conv_layer_for_comb,
                 conv_intermediate_channel_number_for_comb,
                 context_feature_channel,
                 context_agg_layer,
                 mask_interp_neighbor,
                 cls_grid_number,
                 mask_grid_number,
                 interpolation_mode=None,
                 sep_comb_lw_branch=True,
                 context_sep_branch=True,
                 # freeze_solov2_and_train_combonly=False,
                 optimize_list=None,
                 lw_kernel='3x3',
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.4,
                 num_grids=None,
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(AggMaskStarHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.sep_comb_lw_branch = sep_comb_lw_branch
        self.context_sep_branch = context_sep_branch
        self._init_layers()

        self.pre_nms_thresh=0.1

        ##add conv comb layers
        self.optimize_list=optimize_list

        self.cls_grid_number = cls_grid_number
        self.mask_grid_number = mask_grid_number
        self.interpolation_mode = interpolation_mode
        self.nearest_neighbour = mask_interp_neighbor
        # if self.cls_grid_number==self.mask_grid_number and (mask_interp_neighbor==4):
        #     ## for equal grid number of cls and mask branch, and grid number of 4, add one sample point of center itself
        #     self.nearest_neighbour = mask_interp_neighbor+1
        if mask_interp_neighbor == 4:
            if any([clsg == maskg for i, (clsg, maskg) in enumerate(zip(self.cls_grid_number, self.mask_grid_number))]) or interpolation_mode == 5:
                self.nearest_neighbour = mask_interp_neighbor + 1

        self.conv_kernel_for_comb=conv_kernel_for_comb
        self.conv_layer_for_comb=conv_layer_for_comb
        self.conv_intermediate_channel_number_for_comb=conv_intermediate_channel_number_for_comb
        self.context_feature_channel = context_feature_channel

        learned_weight_number_per_layer={}
        learned_weight_kernel_shape = {}
        for lr in range(self.conv_layer_for_comb):
            if lr==0:#first layer
                if self.conv_layer_for_comb==1:## only one layer, so output channel number is 1
                    w = 1 * (self.nearest_neighbour+self.context_feature_channel) * self.conv_kernel_for_comb * self.conv_kernel_for_comb
                    b = 1
                    learned_weight_number_per_layer[lr]=[w, b]
                    learned_weight_kernel_shape[lr]=[1, (self.nearest_neighbour+self.context_feature_channel), self.conv_kernel_for_comb, self.conv_kernel_for_comb]
                else:
                    w = self.conv_intermediate_channel_number_for_comb * (self.nearest_neighbour+self.context_feature_channel) * self.conv_kernel_for_comb * self.conv_kernel_for_comb
                    b = self.conv_intermediate_channel_number_for_comb
                    learned_weight_number_per_layer[lr]=[w, b]
                    learned_weight_kernel_shape[lr]=[self.conv_intermediate_channel_number_for_comb, (self.nearest_neighbour+self.context_feature_channel), self.conv_kernel_for_comb, self.conv_kernel_for_comb]
            elif lr==self.conv_layer_for_comb-1:# last layer
                w = 1 * self.conv_intermediate_channel_number_for_comb * self.conv_kernel_for_comb * self.conv_kernel_for_comb
                b = 1
                learned_weight_number_per_layer[lr]=[w, b]
                learned_weight_kernel_shape[lr] = [1,self.conv_intermediate_channel_number_for_comb,
                                                   self.conv_kernel_for_comb,
                                                   self.conv_kernel_for_comb]
            else:# intermediate layer
                w = self.conv_intermediate_channel_number_for_comb * self.conv_intermediate_channel_number_for_comb * self.conv_kernel_for_comb * self.conv_kernel_for_comb
                b = self.conv_intermediate_channel_number_for_comb
                learned_weight_number_per_layer[lr] = [w, b]
                learned_weight_kernel_shape[lr] = [self.conv_intermediate_channel_number_for_comb,self.conv_intermediate_channel_number_for_comb,
                                                   self.conv_kernel_for_comb,
                                                   self.conv_kernel_for_comb]

        self.learned_weight_number_per_layer = learned_weight_number_per_layer
        self.learned_weight_number=sum([sum(item[1]) for item in learned_weight_number_per_layer.items()])
        self.learned_weight_kernel_shape= learned_weight_kernel_shape

        if lw_kernel =='3x3':
            self.learned_weights = nn.Conv2d(
                in_channels, self.learned_weight_number, kernel_size=3, stride=1,
                padding=1
            )
        elif lw_kernel =='1x1':
            self.learned_weights = nn.Conv2d(
                in_channels, self.learned_weight_number, kernel_size=1, stride=1,
                padding=0
            )

        if self.context_feature_channel>0:
            self.context_fusion_convs = []
            if self.context_sep_branch:

                if context_agg_layer == 1:
                    self.context_fusion_convs.append(nn.Conv2d(in_channels * 5, self.context_feature_channel, 1))
                    self.context_fusion_convs.append(nn.ReLU())
                elif context_agg_layer == 2:
                    self.context_fusion_convs.append(nn.Conv2d(in_channels * 5, in_channels, 1))
                    self.context_fusion_convs.append(nn.GroupNorm(32, in_channels))
                    self.context_fusion_convs.append(nn.ReLU())
                    self.context_fusion_convs.append(nn.Conv2d(in_channels, self.context_feature_channel, 1))
                    self.context_fusion_convs.append(nn.ReLU())

            else:
                self.context_fusion_convs.append(nn.Conv2d(in_channels, self.context_feature_channel, 1))
                self.context_fusion_convs.append(nn.ReLU())
            self.context_fusion_convs = nn.Sequential(*self.context_fusion_convs)

    def fuse_context_feature(self, x):
        context_target_feature_size = x[0].shape[2:]
        context_features=[]
        for l in range(len(x)):
            context_features.append(F.upsample_bilinear(x[l], context_target_feature_size))
        return self.context_fusion_convs(torch.cat(context_features,dim=1))

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.feature_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        if self.sep_comb_lw_branch:
            self.kernel_convs_convcomb=nn.ModuleList()

        # mask feature
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.feature_convs.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    if i == 3:
                        in_channel = self.in_channels + 2
                    else:
                        in_channel = self.in_channels
                    one_conv = ConvModule(
                        in_channel,
                        self.seg_feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=norm_cfg is None)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue
                one_conv = ConvModule(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)
            self.feature_convs.append(convs_per_level)

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels

            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            if self.sep_comb_lw_branch:
                self.kernel_convs_convcomb.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0)
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
        self.solo_mask = ConvModule(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0, norm_cfg=norm_cfg, bias=norm_cfg is None)

    def init_weights(self):
        # TODO: init for feat_conv
        for m in self.feature_convs:
            s = len(m)
            for i in range(s):
                if i % 2 == 0:
                    normal_init(m[i].conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        if self.sep_comb_lw_branch:
            for m in self.kernel_convs_convcomb:
                normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (feats[0].shape[-2], feats[0].shape[-1])
        kernel_pred, cate_pred, spatial_index, learned_upsample_weights = multi_apply(self.forward_single, new_feats,
                                             list(range(len(self.cls_grid_number))),
                                             eval=eval)
        # add coord for p5
        x_range = torch.linspace(-1, 1, feats[-2].shape[-1], device=feats[-2].device)
        y_range = torch.linspace(-1, 1, feats[-2].shape[-2], device=feats[-2].device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feats[-2].shape[0], 1, -1, -1])
        x = x.expand([feats[-2].shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feature_add_all_level = self.feature_convs[0](feats[0])
        for i in range(1, 3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](torch.cat([feats[3], coord_feat], 1))

        feature_pred = self.solo_mask(feature_add_all_level)
        feature_pred_clone = feature_pred.clone()
        N, c, h, w = feature_pred.shape
        feature_pred = feature_pred.view(-1, h, w).unsqueeze(0)
        ins_pred = []

        for i in range(5):
            kernel = kernel_pred[i].permute(0, 2, 3, 1).contiguous().view(-1, c).unsqueeze(-1).unsqueeze(-1)
            ins_i = F.conv2d(feature_pred, kernel, groups=N).view(N, self.mask_grid_number[i] ** 2, h, w)
            # if not eval:
            #     ins_i = F.interpolate(ins_i, size=(featmap_sizes[i][0] * 2, featmap_sizes[i][1] * 2), mode='bilinear')
            # if eval:
            #     ins_i = ins_i.sigmoid()
            ins_pred.append(ins_i)

        ##add context feature
        if self.context_feature_channel > 0:
            if self.context_sep_branch:
                context_feature = self.fuse_context_feature(feats)
            else:
                context_feature = self.context_fusion_convs(feature_pred_clone)
        else:
            context_feature = []
        return ins_pred, cate_pred, spatial_index, learned_upsample_weights, context_feature, upsampled_size

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False):
        kernel_feat = x
        kernel_feat_conv_comb = x
        cate_feat = x
        # kernel branch
        # concat coord

        x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device)
        y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        kernel_feat = torch.cat([kernel_feat, coord_feat], 1)
        for i, kernel_layer in enumerate(self.kernel_convs):
            if i == self.cate_down_pos:
                mask_num_grid = self.mask_grid_number[idx]
                kernel_feat = F.interpolate(kernel_feat, size=mask_num_grid, mode='bilinear')
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                cls_num_grid = self.cls_grid_number[idx]
                cate_feat = F.interpolate(cate_feat, size=cls_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        ##seperate branch for predicting conv comb weights
        if self.sep_comb_lw_branch:
            kernel_feat = torch.cat([kernel_feat_conv_comb, coord_feat], 1)
            for i, kernel_layer in enumerate(self.kernel_convs_convcomb):
                if i == self.cate_down_pos:
                    cls_num_grid = self.cls_grid_number[idx]
                    kernel_feat = F.interpolate(kernel_feat, size=cls_num_grid, mode='bilinear')
                kernel_feat = kernel_layer(kernel_feat)
            learned_upsample_weight = self.learned_weights(kernel_feat)
        else:
            learned_upsample_weight = self.learned_weights(kernel_feat)
        ##add spatial index
        l = idx
        shifts_x = torch.arange(
            0, self.cls_grid_number[l], step=1,
            dtype=torch.float32, device=cate_feat.device
        )
        shifts_y = torch.arange(
            0, self.cls_grid_number[l], step=1,
            dtype=torch.float32, device=cate_feat.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        # locations_HW = torch.stack([shift_x, shift_y],dim=2)## transposed index
        locations_HW = torch.stack([shift_y, shift_x], dim=2)  ## correct index
        locations_HW *= ((self.mask_grid_number[l] - 1) / (self.cls_grid_number[l] - 1))
        locations_map_to_HW = locations_HW.view(-1, 2)

        if self.nearest_neighbour == 4 or self.nearest_neighbour == 5 or self.nearest_neighbour == 1 or self.nearest_neighbour == 0:
            if any([clsg == maskg for i, (clsg, maskg) in enumerate(zip(self.cls_grid_number, self.mask_grid_number))]) or self.interpolation_mode==5:  # if equal, get the near 5 points
                ## find nearest center
                neareast_center = locations_map_to_HW.round()

                ## generating nearest neighbour offsets
                shifts_x = torch.arange(
                    -1, 2, step=1,
                    dtype=torch.float32, device=cate_feat.device
                )
                shifts_y = torch.arange(
                    -1, 2, step=1,
                    dtype=torch.float32, device=cate_feat.device
                )
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
                locations_offset = torch.stack([shift_y, shift_x], dim=2)
                locations_offset = torch.stack([locations_offset[0, 1],
                                                locations_offset[1, 0],
                                                locations_offset[1, 1],
                                                locations_offset[1, 2],
                                                locations_offset[2, 1]])
                spatial_index_this_lvl = neareast_center.unsqueeze(1) + locations_offset.unsqueeze(0)
                spatial_index_this_lvl = spatial_index_this_lvl.clamp(0, self.mask_grid_number[l] - 1).long()

            else:  # mask_grid_number less than cls grid number
                right_top_inds = torch.stack([locations_map_to_HW[:, 0].floor(), locations_map_to_HW[:, 1].ceil()],
                                             dim=1).long()
                right_bottom_inds = locations_map_to_HW.ceil().long()
                left_top_inds = torch.stack([locations_map_to_HW[:, 0].ceil(), locations_map_to_HW[:, 1].floor()],
                                            dim=1).long()
                left_bottom_inds = locations_map_to_HW.floor().long()

                spatial_index_this_lvl = torch.stack(
                    [right_top_inds, right_bottom_inds, left_top_inds, left_bottom_inds], dim=1)

        elif self.nearest_neighbour == 9:
            ## find nearest center
            neareast_center = locations_map_to_HW.round()
            ## generating nearest neighbour offsets
            shifts_x = torch.arange(
                -1, 2, step=1,
                dtype=torch.float32, device=cate_feat.device
            )
            shifts_y = torch.arange(
                -1, 2, step=1,
                dtype=torch.float32, device=cate_feat.device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            locations_offset = torch.stack([shift_y, shift_x], dim=2).view(9, 2)
            spatial_index_this_lvl = neareast_center.unsqueeze(1) + locations_offset.unsqueeze(0)
            spatial_index_this_lvl = spatial_index_this_lvl.clamp(0, self.mask_grid_number[
                l] - 1).long()  ## clamp for boundary points

        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return kernel_pred, cate_pred, spatial_index_this_lvl, learned_upsample_weight

    def loss(self,
             ins_preds,
             cate_preds,
             spatial_index,
             learned_upsample_weights,
             context_feature,
             upsample_sizes,
             learned_weight_kernel_shape,
             learned_weight_number_wb,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)
        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        # ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
        #                         for ins_preds_level_img, ins_ind_labels_level_img in
        #                         zip(ins_preds_level, ins_ind_labels_level)], 0)
        #              for ins_preds_level, ins_ind_labels_level in zip(ins_preds, zip(*ins_ind_label_list))]
        ins_preds_positive=[]

        ##added
        ins_ind_label_batch_first=[]
        for level in range(len(featmap_sizes)):
            ins_ind_label_batch_first.append(
                torch.stack([ins_ind_label[level] for ins_ind_label in ins_ind_label_list], dim=0)
            )

        N = len(ins_ind_label_list)
        interp_neighbour = spatial_index[0].size(1)
        for l in range(len(featmap_sizes)):
            S = self.mask_grid_number[l]
            mask_regression_this_lvl=ins_preds[l]
            _, _, H, W = mask_regression_this_lvl.shape

            locations_gt_inds_batch_index = ins_ind_label_batch_first[l].clone().long()
            if (locations_gt_inds_batch_index != 0).sum() > 0:
                locations_gt_inds_batch_index[locations_gt_inds_batch_index==0]=-1
                for ll in range(len(locations_gt_inds_batch_index)):
                    locations_gt_inds_batch_index[ll][locations_gt_inds_batch_index[ll] != -1] = ll
                spatial_index_this_lvl_pos_batch_index = locations_gt_inds_batch_index[locations_gt_inds_batch_index != -1]

                spatial_index_this_lvl = spatial_index[l]
                learned_upsample_weights_this_lvl=learned_upsample_weights[l]
                spatial_index_this_lvl_pos = spatial_index_this_lvl[None].expand(N, -1, -1, -1)[ins_ind_label_batch_first[l] != False]
                learned_upsample_weights_this_lvl_pos = \
                learned_upsample_weights_this_lvl.view(N, learned_upsample_weights_this_lvl.size()[1], -1).permute(0, 2, 1)[ins_ind_label_batch_first[l] != False]
                if len(context_feature)>0:
                    context_feature_this_lvl_pos = F.upsample_bilinear(context_feature[spatial_index_this_lvl_pos_batch_index], (H, W))
                spatial_index_this_lvl_pos_batch_index = spatial_index_this_lvl_pos_batch_index[:, None].expand(-1,interp_neighbour).contiguous().view(-1)

                sampled_masks_before_weighted_comb = mask_regression_this_lvl.view(N, S, S, H, W)[
                    spatial_index_this_lvl_pos_batch_index,
                    spatial_index_this_lvl_pos.view(-1, 2)[:, 0],
                    spatial_index_this_lvl_pos.view(-1, 2)[:, 1]]


                combined_masks = sampled_masks_before_weighted_comb.view(-1, interp_neighbour, H, W)
                if self.nearest_neighbour==1:
                    combined_masks = combined_masks[:, 2:3, ...]
                    ## concat context feature
                    if len(context_feature) > 0:
                        combined_masks = torch.cat([combined_masks, context_feature_this_lvl_pos], dim=1)

                elif self.nearest_neighbour==0:
                    combined_masks=context_feature_this_lvl_pos

                else:
                    if len(context_feature) > 0:
                        combined_masks = torch.cat([combined_masks, context_feature_this_lvl_pos], dim=1)

                indexing = 0
                for comb_layer in range(len(learned_weight_kernel_shape)):
                    w_shape = learned_weight_kernel_shape[comb_layer].copy()
                    w_shape[0] *= learned_upsample_weights_this_lvl_pos.size(
                        0)  ## arange into group conv, refer to F.conv2d argument requirement
                    w_number = learned_weight_number_wb[comb_layer][0]
                    conv_w = learned_upsample_weights_this_lvl_pos[:, indexing:(indexing + w_number)].contiguous().view(
                        w_shape)

                    indexing += w_number


                    b_number = learned_weight_number_wb[comb_layer][1]
                    conv_b = learned_upsample_weights_this_lvl_pos[:, indexing:(indexing + b_number)].contiguous().view(-1)

                    indexing += b_number

                    padding = int(w_shape[-1] / 3)
                    groups = combined_masks.size(0)

                    if comb_layer == 0:
                        if len(learned_weight_kernel_shape) == 1:
                            conv_comb = F.conv2d(combined_masks.view(-1, H, W)[None], conv_w, conv_b, padding=padding,
                                                 groups=groups)
                        else:
                            conv_comb = F.conv2d(combined_masks.view(-1, H, W)[None], conv_w, conv_b, padding=padding,
                                                 groups=groups)
                            conv_comb = F.relu(conv_comb)
                    elif comb_layer == (len(learned_weight_kernel_shape) - 1):
                        conv_comb = F.conv2d(conv_comb, conv_w, conv_b, padding=padding, groups=groups)
                    else:
                        conv_comb = F.conv2d(conv_comb, conv_w, conv_b, padding=padding, groups=groups)
                        conv_comb = F.relu(conv_comb)

                ins_preds_positive.append(conv_comb.squeeze(0))
                # ins_preds_positive.append(center_masks)

            else:
                ins_preds_positive.append([])

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.int().sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds_positive, ins_labels):
            # if input.size()[0] == 0:
            if len(input) == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate)

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.cls_grid_number):
            stride=self.strides[0]
            featmap_size=featmap_sizes[0]
            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                    continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def forward_for_single_feature_map(
            self, box_cls,
            box_regression, level, spatial_index, learned_weights, S, learned_weight_kernel_shape,learned_weight_number_wb, context_feature, upsampled_size, mask_thr):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, H, W, C = box_cls.shape

        def _nms(heat, kernel=3, prenms_thr=0.2):
            pad = (kernel - 1) // 2

            hmax = torch.nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()*(heat>prenms_thr).float()
            return heat * keep

        # box_cls = box_cls.sigmoid()
        # if self.inference_maxpooling_nms:
        #     box_cls = _nms(box_cls)

        # put in the same format as locations
        # box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C)
        # box_regression = box_regression.view(N, -1, H, W).permute(0, 2, 3, 1)
        # box_regression = box_regression.reshape(N, -1, 4)
        # centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh

        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # box_cls = box_cls * centerness[:, :, None]



        mask_nms_strides = [1, 2, 4, 8, 16]
        results = []
        for i in range(N):

            if candidate_inds.view(N, -1).sum(1)[i].item() == 0:
                results.append(None)
                continue

            per_img_box_cls = box_cls[i]
            per_img_candidate_inds = candidate_inds[i]
            per_img_box_cls_candidate_score = per_img_box_cls[per_img_candidate_inds]

            ## get the interpolated mask
                ## original
            # per_img_mask_pred = box_regression[i][per_img_candidate_inds.nonzero()[:,0]]## cls agnostic
            pos_inds = per_img_candidate_inds.nonzero()[:,0]
            learned_weight_this_batch = learned_weights[i]
            mask_H, mask_W = box_regression.shape[2:]
            if self.context_feature_channel > 0:
                context_feature_this_batch = F.upsample_bilinear(context_feature[i][None], (mask_H, mask_W)).expand(pos_inds.size()[0],-1,-1,-1)
            interp_neighbour = spatial_index.size(1)


            pos_spatial_index = spatial_index[pos_inds].view(-1,2)
            sampled_masks = box_regression[i].view(S, S, mask_H, mask_W)[None][
                                [0] * pos_spatial_index.shape[0], pos_spatial_index[:, 0], pos_spatial_index[:, 1]]
            sampled_masks= sampled_masks.view(-1, interp_neighbour, mask_H, mask_W)

            weigths_sampled_masks = learned_weight_this_batch.view(learned_weight_this_batch.size(0),-1).permute(1,0)[pos_inds]

            if self.nearest_neighbour == 1:
                sampled_masks = sampled_masks[:, 2:3, ...]

                ## concat context feature
                if self.context_feature_channel > 0:
                    sampled_masks = torch.cat([sampled_masks, context_feature_this_batch], dim=1)

            elif self.nearest_neighbour==0:
                sampled_masks=context_feature_this_batch

            else:
                if self.context_feature_channel > 0:
                    sampled_masks = torch.cat([sampled_masks, context_feature_this_batch], dim=1)

            indexing = 0
            # try:

            for comb_layer in range(len(learned_weight_kernel_shape)):
                w_shape = learned_weight_kernel_shape[comb_layer].copy()
                w_shape[0] *= weigths_sampled_masks.size(
                    0)  ## arange into group conv, refer to F.conv2d argument requirement
                w_number = learned_weight_number_wb[comb_layer][0]
                conv_w = weigths_sampled_masks[:, indexing:(indexing + w_number)].contiguous().view(w_shape)

                indexing += w_number

                b_number = learned_weight_number_wb[comb_layer][1]
                conv_b = weigths_sampled_masks[:, indexing:(indexing + b_number)].contiguous().view(-1)

                indexing += b_number

                padding = int(w_shape[-1] / 3)
                groups = sampled_masks.size(0)


                if comb_layer == 0:
                    if len(learned_weight_kernel_shape)==1:
                        conv_comb = F.conv2d(sampled_masks.view(-1, mask_H, mask_W)[None], conv_w, conv_b, padding=padding, groups=groups)
                    else:
                        conv_comb = F.conv2d(sampled_masks.view(-1, mask_H, mask_W)[None], conv_w, conv_b, padding=padding, groups=groups)
                        conv_comb = F.relu(conv_comb)
                elif comb_layer == (len(learned_weight_kernel_shape) - 1):
                    conv_comb = F.conv2d(conv_comb, conv_w, conv_b, padding=padding, groups=groups)
                else:
                    conv_comb = F.conv2d(conv_comb, conv_w, conv_b, padding=padding, groups=groups)
                    conv_comb = F.relu(conv_comb)

            # weighted_masks=conv_comb.squeeze(0)
            # weighted_masks = (sampled_masks.view(-1, interp_neighbour, mask_H, mask_W)*weigths_sampled_masks[:,:,None][:,:,None]).sum(1)
            # per_img_mask_pred = weighted_masks


            # per_img_box_cls_candidate_class = per_img_candidate_inds.nonzero()[:, 1]+1
            per_img_box_cls_candidate_class = per_img_candidate_inds.nonzero()[:, 1]


            # except:
            #     print('df')

            # if per_img_pre_nms_top_n.item()==0:
            #     results.append(None)
            #     continue
            #
            # if per_img_box_cls_candidate_score.size(0) > per_img_pre_nms_top_n.item():
            #     per_img_box_cls_candidate_score, topk_inds = \
            #         per_img_box_cls_candidate_score.topk(per_img_pre_nms_top_n, sorted=False)
            #     per_img_mask_pred = per_img_mask_pred[topk_inds]
            #     per_img_box_cls_candidate_class = per_img_box_cls_candidate_class[topk_inds]

            # h, w = image_sizes[i]

            # per_img_mask_pred = torch.nn.functional.interpolate(
            #                         input=per_img_mask_pred[None],
            #                         # scale_factor=mask_nms_strides[level],
            #                         size = mask_nms_size,
            #                         mode="bilinear", align_corners=True)
            # per_img_mask_pred = F.interpolate(conv_comb.sigmoid(), size=upsampled_size, mode='bilinear').squeeze(0)
            per_img_mask_pred = conv_comb.sigmoid().squeeze(0)

            seg_masks = per_img_mask_pred > mask_thr
            sum_masks = seg_masks.sum((1, 2)).float()

            keep = sum_masks > self.strides[level]
            if keep.sum() == 0:
                results.append(None)
                continue

            results.append((per_img_box_cls_candidate_score[keep], per_img_mask_pred[keep, ...], per_img_box_cls_candidate_class[keep]))

        return results

    def cat_masklist(self, masks):
        cated_cls_scores = torch.cat(
            [per_level_pred[0] for per_level_pred in list(masks) if per_level_pred is not None])
        cated_masks = torch.cat([per_level_pred[1] for per_level_pred in list(masks) if per_level_pred is not None])
        cated_cls_class = torch.cat([per_level_pred[2] for per_level_pred in list(masks) if per_level_pred is not None])

        return cated_cls_scores, cated_masks, cated_cls_class


    def get_seg(self, seg_preds, cate_preds,
                spatial_index,
                learned_upsample_weights,
                context_feature,
                upsampled_size,
                learned_weight_kernel_shape,
                learned_weight_number_wb,
                img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        ## add conv comb，
        sampled_boxes=[]
        for i, (o, b, spi, lw, S) in enumerate(zip(cate_preds, seg_preds, spatial_index,
                                                         learned_upsample_weights, self.mask_grid_number)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    o, b, i, spi, lw, S, learned_weight_kernel_shape,learned_weight_number_wb, context_feature, upsampled_size, cfg.mask_thr
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [self.cat_masklist(boxlist) if boxlist!=(None,None,None,None,None) else None for boxlist in boxlists ]

        result_list = []
        for img_id in range(len(img_metas)):
            # cate_pred_list = [
            #     cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            # ]
            # seg_pred_list = [
            #     seg_preds[i][img_id].detach() for i in range(num_levels)
            # ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            if boxlists[img_id]==None:
                result_list.append(None)
                continue
            cate_pred_list = boxlists[img_id][0]
            seg_pred_list = boxlists[img_id][1]
            cate_labels = boxlists[img_id][2]

            result = self.get_seg_single_threshed(cate_pred_list, seg_pred_list, cate_labels,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single_threshed(self,
                       cate_preds,
                       seg_preds,
                       cate_labels,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        # inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds
        if len(cate_scores) == 0:
            return None

        # masks.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # mask scoring. ## average confidence on mask area
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
