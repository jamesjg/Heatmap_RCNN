_base_ = '../default.py'

data = dict(
    data_type = 'WFLW',
    num_landmarks = 98,
    data_folder = 'train_with_box',
    test_folder = 'test_with_box'
)

model = dict(
    backbone= 'resnet50',
    use_fpn = True,

    output_stages=[1,2,3,4], # correspond to 64,32,16,8
    sup_losses = ['REG', 'REG', 'REG', 'REG'], # supervised Loss
    multi_stage_sigmas = [1.33, 1.33, 1.33, 1.0], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
    roi_flat_sizes = [7,5,5,5], # 7x7 roi for 64x64 heatmap, 5x5 for 32x32...

    roi_loss = "L2",
)

# model = dict(
#     backbone= 'resnet50',
#     use_fpn = True,

#     output_stages=[2,3,4], # correspond to 64,32,16,8
#     sup_losses = [ 'REG', 'REG', 'REG'], # supervised Loss
#     multi_stage_sigmas = [ 1.33, 1.33, 1.0], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
#     roi_flat_sizes = [7,5,5], # 7x7 roi for 64x64 heatmap, 5x5 for 32x32...

#     roi_loss = "L2",
# )

# model = dict(
#     backbone= 'resnet50',
#     use_fpn = True,

#     output_stages=[3], # correspond to 64,32,16,8
#     sup_losses = ['REG'], # supervised Loss
#     multi_stage_sigmas = [1.0], # heatmap sigma for multi stage heatmaps, 1.33 for 64x64...
#     roi_flat_sizes = [7], # 7x7 roi for 64x64 heatmap, 5x5 for 32x32...

#     roi_loss = "L2",
# )


train = dict(
    num_epochs = 120,
    scheduler = "ReduceLROnPlateau",
    decay_steps=[30,60,90],
    num_workers=4,
    init_lr=1e-5
)