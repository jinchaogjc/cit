_base_ = [
        '../r50-D8_ade20k_dv3_cl.py',
]
crop_size = (512, 512)
pre_num_cls = 100
new_num_cls = 50
train_class_range = [101, 150]
test_class_range = [1, 150]
num_cls = train_class_range[-1] - train_class_range[0] + 1
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoderCL',
    pre_num_cls=pre_num_cls,
    new_num_cls=new_num_cls,
    cls_factor=10.0,
    attn_factor=20.0,
    decode_head=dict(
        num_classes=pre_num_cls+new_num_cls,
        loss_decode=dict(
                    _delete_=True,type='ATMLoss', num_classes=1, dec_layers=1, 
                    loss_weight=1.0, ignore_range=pre_num_cls)
                ),
    )

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
             window_size=50),
        dict(data_src='aux.acc_seg',
             method_name='mean',
            #  log_name='acc_seg_large_window',
             window_size=50)
    ],
)
find_unused_parameters = True