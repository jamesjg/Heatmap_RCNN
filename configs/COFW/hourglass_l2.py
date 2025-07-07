_base_ = '../default.py'

data = dict(
    data_type = 'COFW',
    num_landmarks = 29,
    data_folder = 'train',
    test_folder = 'test'
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