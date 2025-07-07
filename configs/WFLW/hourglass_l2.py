_base_ = '../default.py'

data = dict(
    data_type = 'WFLW',
    num_landmarks = 98,
    data_folder = 'train_with_box',
    test_folder = 'test_with_box'
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
    scheduler = "ReduceLROnPlateau",
    decay_steps=[30,60,90],
    init_lr=1e-5
)