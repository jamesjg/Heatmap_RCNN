_base_ = '../default.py'

data = dict(
    data_type = 'WFLW',
    num_landmarks = 98,
    data_folder = 'train_with_box',
    test_folder = 'test_with_box'
)

model = dict(
    backbone = 'hourglass',
    loss = 'l2',

    use_roi = True,
    roi_feature = ["heatmap"], # heatmap, backbone, multi_heatmaps
    roi_size = 7,
    roi_loss = "L2",
)


backbone = dict(
    net_name = 'hourglass',
    num_stack = 4,
    num_layer = 4,
    num_feature = 256,
    sel_indices = [0,1,2,3] # 对哪次stack进行监督, 最大值需要小于num_stack
)

train = dict(
    num_epochs = 120,
    scheduler = "MultiStepLR",
    decay_steps=[30,55,80],
    init_lr=1e-5,
    roi_init_lr=1e-2
)
