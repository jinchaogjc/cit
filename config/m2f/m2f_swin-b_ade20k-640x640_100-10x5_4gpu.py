_base_ = [
    './m2f_swin-b_ade20k-640x640_100.py'
]

crop_size = (640, 640)
init_num_class = 100
increment = 10
dataset_total_cls = 150
max_iters = 80000

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
num_queries = init_num_class
model = dict(
    type='EncoderDecoderCL_m2f',
    pre_num_cls=init_num_class,
    new_num_cls=increment,
    cls_factor=10.0,
    attn_factor=30.0,
    decode_head=dict(
        type='M2fHead',
        num_queries=num_queries,
        loss_cls_ignore_range=init_num_class,
        ),
    )

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=train_class_range, anno_check=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=test_class_range),
    dict(type='PackSegInputs')
]
val_dataloader = dict(batch_size=1,
                      dataset=dict(pipeline=test_pipeline),)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(
    type='ScenarioBasedTrainLoopM2f', scenario=scenario, max_iters=max_iters, val_interval=2000)
val_cfg = dict(type='ScenarioBasedValLoop', scenario=scenario)
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=16000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

find_unused_parameters = True