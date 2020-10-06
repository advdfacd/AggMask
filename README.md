
#AggMask: Exploring aggregated learning of local mask representations for high quality instance segmentation

This is an anonymous repository containing the instructions and pretrained model of the ICLR submission (ID: 189):

## Installation 
This code is based on 
[mmdetection v1.0.0](https://github.com/open-mmlab/mmdetection). Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.
Or run the following installation script:
###
    conda create -n aggmask_mmdet python=3.7
    source activate aggmask_mmdet
    echo "python path"
    which python
    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
    pip install cython==0.29.12 mmcv==0.2.16 matplotlib terminaltables
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
    pip install opencv-python-headless
    pip install Pillow==6.1
    pip install -v -e .
    
    mkdir data
    ln -s $COCO_ROOT data
For [LVIS](https://www.lvisdataset.org/dataset) dataset, only lvis validation set is needed, please arrange the data as:

```
mmdetection
├── configs
├── data
│   ├── LVIS
│   │   ├── lvis_v0.5_val_cocofied.json
│   │   ├── images
│   │   │   ├── val2017
```
> lvis_v0.5_val_cocofied.json is the annotation COCO category subset of LVIS validation set, you can download our processed file at [GoogleDrive](https://drive.google.com/file/d/1Qjb44bIJokIUp677NhkTb3d1OXLNZfs8/view?usp=sharing)
or prepare it by [prepare_cocofied_lvis](https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_cocofied_lvis.py
)

>note for  LVIS images, you can just create a softlink for the val2017 to point to COCO val2017

## Config Files
Please use corresponding config files for training, evaluation or visualization

Model | config file
--- |:---:
AggMask_R50_FPN   | aggmask_r50_fpn.py
AggMask_R101_FPN  | aggmask_r101_fpn.py
AggMask*_R50_FPN  | aggmask_star_r50_fpn.py
AggMask*_R101_FPN | aggmask_star_r101_fpn.py
AggMask_R101_FPN +cls-grid  | aggmask_r101_fpn_increasing_clsgrid.py
AggMask_R101_FPN -mask-grid | aggmask_r101_fpn_halve_maskgrid.py
AggMask*_R101_FPN +cls-grid  | aggmask_star_r101_fpn_increasing_clsgrid.py
AggMask*_R101_FPN -mask-grid | aggmask_star_r101_fpn_halve_maskgrid.py
> config files under ./configs/aggmask/

> ##AggMask is with SOLO, and AggMask* is with SOLOv2##

## Training (with multiple GPUs)
    python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$((RANDOM + 10000)) tools/train.py ${CONFIG_FILE} --launcher pytorch

    Example (8 gpus): 
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((RANDOM + 10000)) tools/train.py ./configs/aggmask/aggmask_r101_fpn_halve_maskgrid.py --launcher pytorch

## Evaluation on COCO minival and LVIS val:
    python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    python tools/test_ins.py ./configs/aggmask/aggmask_r101_fpn_halve_maskgrid.py ./aggmask_r101_fpn_halve_maskgrid.pth --show --out aggmask_r101_fpn_halve_maskgrid.pkl --eval segm

> two consecutive evaluation will be performed on COCO minival set ( with COCO api) and 80 COCO category subset of LVIS val set (with LVIS api).

## Evaluation on COCO test-dev:
To evaluate models on COCO test-dev split, please replace the test data by modifying the config file with:
###
     test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
and generate the segmentation json file by:

    python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    python tools/test_ins.py ./configs/aggmask/aggmask_r101_fpn_halve_maskgrid.py ./aggmask_r101_fpn_halve_maskgrid.pth --show --out aggmask_r101_fpn_halve_maskgrid.pkl --eval segm
Then zip the `.pkl.segm.json` to `.pkl.segm.json.zip` and upload to [CodaLab](https://competitions.codalab.org/competitions/20796)


## Pre-trained Models

Model | AP (COCO minival) | AP* (LVIS val) | Link
--- |:---:|:---:|:---:
AggMask_R50_FPN   | 37.9 | 40.0 | [Googledrive](https://drive.google.com/file/d/1o0j_FhMUC6ZlJ9sw72aPGh8wHEBcIelQ/view?usp=sharing)
AggMask_R101_FPN  | 38.6 | 41.2 | [Googledrive](https://drive.google.com/file/d/1PKIva4Cpnk6kMrFoXaEQR_QNMo2SaDwk/view?usp=sharing)
AggMask*_R50_FPN  | 38.7 | 41.0 | [Googledrive](https://drive.google.com/file/d/1dI1sGpIEcU3Jda0S5/view?usp=sharing)
AggMask*_R101_FPN | 39.4 | 42.3 | [Googledrive](https://drive.google.com/file/d/1MnvjhRqpRoWMCUgbPi6a50T1rBfdT6Op/view?usp=sharing)
AggMask_R101_FPN +cls-grid   | 39.1 | 41.6 | [Googledrive](https://drive.google.com/file/d/1dI1sGpIBGX-to35xU9HMTzEcU3Jda0S5/view?usp=sharing)
AggMask_R101_FPN -mask-grid | 38.8 | 41.1 | [Googledrive](https://drive.google.com/file/d/1duVzj3oDVue9qfkCdnugaupSB3i5cUfb/view?usp=sharing)
AggMask*_R101_FPN +cls-grid  | 39.7 | 42.8 | [Googledrive](https://drive.google.com/file/d/1K5ZS4YqbODSMvB1-3h3mvIt03SFmcd_a/view?usp=sharing)
AggMask*_R101_FPN -mask-grid | 39.2 | 42.2 | [Googledrive](https://drive.google.com/file/d/15oPQdKL2PrwWkNkt4saopJDOObHs65cX/view?usp=sharing)


Model | AP (COCO test-dev) | Link
--- |:---:|:---:
AggMask_R101_FPN +cls-grid   | 39.5 | [Googledrive](https://drive.google.com/file/d/1dI1sGpIBGX-to35xU9HMTzEcU3Jda0S5/view?usp=sharing)
AggMask*_R101_FPN +cls-grid  | 40.5 | [Googledrive](https://drive.google.com/file/d/1HX9K5_iOJVFE_tcvStKI0cUM7ty9Z805/view?usp=sharing)

> please use aggmask_r101_fpn_increasing_clsgrid.py and aggmask_star_r101_fpn_increasing_clsgrid_more-60-50-36-16-12.py for the two test-dev models, respectively

## Visualization of instance segmentation result
    
    python tools/test_ins_vis.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --save_dir  ${SAVE_DIR}
    
    Example: 
    python tools/test_ins_vis.py ./configs/aggmask/aggmask_r101_fpn_halve_maskgrid.py  aggmask_r101_fpn_halve_maskgrid.pth --show --save_dir work_dirs/aggmask_r101_fpn_halve_maskgrid
> images with visualized instance segmentation mask will be under save_dir
