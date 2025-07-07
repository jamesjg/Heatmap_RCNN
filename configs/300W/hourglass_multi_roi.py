_base_ = '../default.py'

data = dict(
    data_type = '300W',
    num_landmarks = 68,
    data_folder = 'train',
    test_folder = 'valid'
)

model = dict(
    # roi_layers = 4,
    backbone = 'hourglass',
    loss = 'AWING', # main loss
    roi_feature = "heatmap", # heatmap, backbone
    use_roi = 2, # 0: none, 1: one roi, 2: multi roi     Note use_roi should equal to len(output_stages)
    output_stages = [1,2], # correspond to 64,32,16,8
    multi_stage_sigmas = [1.0,1.0], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
    sup_losses = ['AWING', 'AWING'], # supervised Loss, the first one is equal to main loss # TODO suit for CLS
    stage_heatmap_weights = [1,0.2],
    roi_sizes = [7,5],
    roi_loss = "L2"
)


backbone = dict(
    net_name = 'hourglass',
    num_stack = 4,
    num_layer = 4,
    num_feature = 256,
)

train = dict(
    num_epochs = 120,
    scheduler = "MultiStepLR",
    decay_steps=[30,60,80],
    init_lr=1e-5,
    roi_init_lr=1e-2,
)

