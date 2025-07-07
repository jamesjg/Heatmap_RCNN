import os
# git branch: hourglass_multi_rois
A100_WFLW_OFF = ["logs/WFLW/hourglass/heatmap_rcnn_T/", "logs/WFLW/hourglass/heatmap_rcnn_S/", "logs/WFLW/hourglass/heatmap_rcnn_B/"]
CUDA_ID = 0
MODEL_DIR = "WFLW"

if __name__ == "__main__":

    cmd = ""

    for dir_path in A100_WFLW_OFF:
        ckpt_path = f"{dir_path}ckpt.pth"
        cfg_path = f"{dir_path}config.py"
        cmd += f"python tools/test.py --config_file {cfg_path} --ckpt {ckpt_path} --gpu_id={CUDA_ID} --model_dir={MODEL_DIR} --test_all && "

    cmd = cmd[:-3] + " & "

    print(cmd)
    os.system(cmd)

