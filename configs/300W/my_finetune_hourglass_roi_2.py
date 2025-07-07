_base_ = '../default.py'

data = dict(
    data_type = '300W',
    num_landmarks = 68,
    data_folder = 'train_with_box',
    test_folder = 'valid_with_box'
)

heatmap = dict(
    heatmap_sigma = 1.0
)

model = dict(
    backbone = 'hourglass',
    loss = 'AWING', # main loss
    # ckpt = "logs/300W/hourglass/awing_64_awing_32_4_stack_retina_300W/4/ckpt.pth",
    low_decode_type = 0,
    embed_offset = False,
    roi_sizes = [7,5],

    roi_feature = "heatmap", # heatmap, backbone
    use_roi = 2, # 0: none, 1: one roi, 2: multi roi     Note use_roi should equal to len(output_stages)
    output_stages = [1,2], # correspond to 64,32,16,8
    multi_stage_sigmas = [1.0, 0.666], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
    sup_losses = ['AWING', 'AWING'], # supervised Loss, the first one is equal to main loss # TODO suit for CLS
    stage_heatmap_weights = [1, 0.1],
    roi_loss = "L1"
)


backbone = dict(
    net_name = 'hourglass',
    num_stack = 4,
    num_layer = 4,
    num_feature = 256,
)

train = dict(
    num_epochs = 30,
    scheduler = "MultiStepLR",
    decay_steps=[10, 20],
    init_lr=1e-5,
    roi_init_lr=1e-2
)
    