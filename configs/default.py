exp_name = None

data = dict(
    data_type = None,
    num_landmarks = None,
    data_folder = 'train',
    test_folder = 'test',
    input_size = 256,
)

heatmap = dict(
    heatmap_size = 64,
    heatmap_method="GAUSS",
    heatmap_sigma = 1.33, # "sigma represents 6*sigma+1 kernel size" sigma sets 0 denotes direct(kernel size equals 1), suggest kernel_size sets odd, sigma sets 0.333(3), 0.666(5), 1(7), 1.333(9).
)

model = dict(
    # head_type = 'cls',
    backbone = None,
    loss = None,
    ckpt = None,
    loss_intern_alpha=1.
)

backbone = dict()

train = dict(
    gpu_id=0,
    seed=666,
    batch_size=32,
    num_workers=4,
    init_lr=1e-4,
    num_epochs=100,
    val_epoch = [[200,1]],
    scheduler = None, # "ReduceLROnPlateau | MultiStepLR |  StepLR"
    decay_steps=None,  # [30,60],
    freq = None, # log per 'freq' iters 
)
