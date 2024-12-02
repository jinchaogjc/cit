_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k_cl.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
train_class_range = [1, 100]
test_class_range = [1, 100]
num_cls = train_class_range[-1] - train_class_range[0] + 1
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
                type='DV3Head',
                num_classes=num_cls,
                image_size=crop_size,
                # dropout_ratio=0.0,
                loss_decode=dict(
                    _delete_=True,type='ATMLoss', num_classes=1, dec_layers=1, loss_weight=1.0)
                ),
    auxiliary_head=None)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=train_class_range),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotationsCL', reduce_zero_label=True, class_range=test_class_range),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(pipeline=train_pipeline),)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline),)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR', 
        eta_min=0.0,
        power=0.9,
        begin=1500,
        end=80000,
        by_epoch=False)
]

log_processor = dict(
    by_epoch=False,
    window_size=50,
    custom_cfg=[
        dict(data_src='decode.acc_seg',
             method_name='mean',
            #  log_name='acc_seg_large_window',
             window_size=50),
        dict(data_src='aux.acc_seg',
             method_name='mean',
            #  log_name='acc_seg_large_window',
             window_size=50)
    ],
)
find_unused_parameters = True