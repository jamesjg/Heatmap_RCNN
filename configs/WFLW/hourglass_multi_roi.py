_base_ = '../default.py'

data = dict(
    data_type = 'WFLW',
    num_landmarks = 98,
    data_folder = 'train_with_box',
    test_folder = 'test_with_box'
)

model = dict(
    backbone = 'hourglass',
    loss = 'AWING', # main loss

    roi_feature = "heatmap", # heatmap, backbone
    use_roi = 3, # 0: none, 1: one roi, 2: multi roi     Note use_roi should equal to len(output_stages)
    output_stages = [1,2,3], # correspond to 64,32,16,8
    multi_stage_sigmas = [1.33,1.0,1.0], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
    sup_losses = ['AWING', 'AWING'], # supervised Loss, the first one is equal to main loss # TODO suit for CLS
    stage_heatmap_weights = [1,0.2,0.1],
    roi_sizes = [7,5,3],
    roi_loss = "L2"
)


backbone = dict(
    net_name = 'hourglass',
    num_stack = 4,
    num_layer = 4,
    num_feature = 256,
    sel_indices = [0,1,2] # 在multi——roi中， last stack单独监督，所以不写最后一个
)

train = dict(
    num_epochs = 120,
    scheduler = "MultiStepLR",
    decay_steps=[30,60,80],
    init_lr=1e-5,
    roi_init_lr=1e-2,
)

