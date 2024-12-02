_base_ = [
        '../segvit_dinov2_cl_1x100_40k.py',
]
crop_size = (518, 518)
init_num_class = 100
increment = 50
dataset_total_cls = 150

train_class_range = [init_num_class + 1, init_num_class+increment] 
test_class_range = [1, init_num_class+increment]
num_cls = train_class_range[-1] - train_class_range[0] + 1

scenario = dict(
    nb_tasks=(dataset_total_cls - init_num_class) // increment,
    increment=increment,
    initial_increment=init_num_class,
    start_task=0,
    num_classes=dataset_total_cls,
)

model = dict(
    type='EncoderDecoderCL',
    pre_num_cls=init_num_class,
    new_num_cls=increment,
    # cls_factor=5.0,
    # attn_factor=30.0,
    backbone=dict(
        freeze=True,
        ),
    decode_head=dict(
        num_classes=init_num_class,
        loss_decode=dict(ignore_range=init_num_class)
                ),
    )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=train_class_range),
    dict(
        type='RandomResize',
        scale=(2048, 518),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 518), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=test_class_range),
    dict(type='PackSegInputs')
]

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=4,
                        dataset=dict(pipeline=train_pipeline),)
val_dataloader = dict(batch_size=1,
                      dataset=dict(pipeline=test_pipeline),)
test_dataloader = val_dataloader

# runtime settings
train_cfg = dict(
    type='ScenarioBasedTrainLoop',
    scenario=scenario
)

val_cfg = dict(
    type='ScenarioBasedValLoop',
    scenario=scenario
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR', 
        eta_min=0.0,
        power=0.9,
        begin=1500,
        end=40000,
        by_epoch=False)
]
log_processor = dict(
    by_epoch=False,
    window_size=50,
    custom_cfg=[
        dict(data_src='decode.acc_seg',
             method_name='mean',
            #  log_name='acc_seg_large_window',
             window_size=50)
    ],
)


find_unused_parameters=True