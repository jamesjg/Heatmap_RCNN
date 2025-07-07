_base_ = '../default.py'

data = dict(
    data_type = '300W',
    num_landmarks = 68,
    data_folder = 'train_with_box',
    test_folder = 'valid_with_box'
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
    decay_steps=[30,60,90],
    init_lr=1e-5,
    roi_init_lr=1e-2
)

model = dict(
    backbone = 'hourglass',
    loss = 'AWING',

    use_roi = True,
    roi_feature = ["heatmap"], # heatmap, backbone, multi_heatmaps
    roi_size = 7,
    roi_loss = "L2",
)

