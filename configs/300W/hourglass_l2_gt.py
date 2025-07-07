_base_ = '../default.py'

data = dict(
    data_type = '300W',
    num_landmarks = 68,
    data_folder = 'train',
    test_folder = 'valid'
)

model = dict(
    backbone = 'hourglass',
    loss = 'l2'
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
    init_lr=1e-5
)