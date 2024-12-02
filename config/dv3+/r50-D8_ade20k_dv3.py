_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
                type='DV3Head',
                num_classes=150,
                image_size=crop_size,
                loss_decode=dict(
                    _delete_=True,type='ATMLoss', num_classes=1, dec_layers=1, loss_weight=1.0)
                ),
    auxiliary_head=None)

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