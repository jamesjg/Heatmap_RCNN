_base_ = '../default.py'

data = dict(
    data_type = 'WFLW',
    num_landmarks = 98,
    data_folder = 'train',
    test_folder = 'test'
)

model = dict(
    backbone = 'hourglass',
    loss = 'AWING',

    use_roi = True,
    roi_feature = ["heatmap"], # heatmap, backbone, multi_heatmaps
    roi_size = 7,
    roi_loss = "L2",
)


backbone = dict(
    net_name = 'hourglass',
    num_stack = 4,
    num_layer = 4,
    num_feature = 256
)

train = dict(
    num_epochs = 10,
    scheduler = "MultiStepLR",
    decay_steps=[5],
    init_lr=1e-5,
    roi_init_lr=1e-3,
)
