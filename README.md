# Heatmap RCNN
This is the official PyTorch code for ```One General Plug-In for Facial Heatmap-based Keypoint Detection```.


## Installation
Install file referenced at `environment.yml`

## Data Preparation
Download the datasets and organize the data as follows:
```
data
├── benchmark
|   ├── 300W
|   |   |—— train
|   |   |—— test
|   |   |—— valid
|   |   |—— valid
|   |   |—— valid
|   |   ├── train.txt
|   |   ├── ...
|   ├── WFLW
|   |   ├── train
|   |   |—— test
|   |   |—— test_blur
|   |   |—— test_expression
|   |   |—— test_illumination
|   |   |—— ...
|   |   ├── train.txt
|   |   ├── ...
|   ├── COFW
|   |   ├── train
|   |   |—— test
|   |   ├── train.txt
|   |   ├── test.txt
where .txt files contain the image names and landmarks.
```
## Test
The CKPTs and configs of Heatmap RCNN for different datasets are stored in [Baidu Disk](https://pan.baidu.com/s/160C7w27iu9Ac46Mf8cMs2Q?pwd=dizh)

Download the logs.zip and unzip it to the repo root.

### Quick Test for 300W, COFW, WFLW
```    
python shells/300W.py, shells/COFW.py, shells/WFLW.py
```
The test results will be saved in the `test_logs` folder.

### Test using the command line
Also, you can test the model by specifying the config file and the ckpt path.
```
python tools/test.py --config_file {cfg_path} --ckpt {ckpt_path} --gpu_id={CUDA_ID}
```
This will test the model on the fullset. To test all subsets, add the `--test_all` flag.

## Train

### train the baseline model
```
python tools/train.py --config_file configs/WFLW/hourglass_multi_roi.py --sup_losses=AWING,AWING,AWING --stage_heatmap_weights=1,{weight_32},{weight_16} --multi_stage_sigmas=1.333,{sigma_32},{sigma_16} --loss=AWING --num_stack=4 --gpu_id={CUDA_ID}  --model_dir=awing_64_awing_32_awing_16_4_stack_gt --init_lr=2e-5 --scheduler=MultiStepLR --heatmap_sigma=1.333 --data_folder=train --test_folder=test
```

### finetune Response-Aware Module 
```
python tools/train.py --config_file configs/WFLW/fine_hourglass_roi_3.py --model_dir=finetune_4_stacks_3_ROI --num_stack=4 --ckpt logs/WFLW/hourglass/awing_64_awing_32_awing_16_4_stack_gt/8/ckpt.pth --gpu_id={CUDA_ID} --roi_sizes=7,5,3 --ft --init_roi_weight --low_decode_type=1 --embed_offset=True 
```
    


## Response-Aware Module

In addition to the baselines, we also implement the RAM to the model zoos, Dark-human-pose and HRNet-face-alginment.

DARK-RAM referenced at https://github.com/starhiking/hrnet-pose-SAM

HRNet-RAM referenced at https://github.com/starhiking/hrnet_roi_fc