import os
# git branch: hourglass_multi_rois
Me_3090_300W_OFF = ["logs/300W/hourglass/heatmap_rcnn_T/", "logs/300W/hourglass/heatmap_rcnn_S/", "logs/300W/hourglass/heatmap_rcnn_B/"]
CUDA_ID = 0
MODEL_DIR = "300W"

if __name__ == "__main__":

    cmd = ""

    for dir_path in Me_3090_300W_OFF:
        ckpt_path = f"{dir_path}ckpt.pth"
        cfg_path = f"{dir_path}config.py"
        cmd += f"python tools/test.py --config_file {cfg_path} --ckpt {ckpt_path} --gpu_id={CUDA_ID} --model_dir={MODEL_DIR} --test_all && "

    cmd = cmd[:-3] + " & "

    print(cmd)
    os.system(cmd)

