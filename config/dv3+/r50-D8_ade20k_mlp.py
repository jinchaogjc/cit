_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='DV3PlusMLPHead',
        num_classes=150,
        image_size=crop_size,
        # loss_decode=dict(
        #         _delete_=True,
        #         type='ATMLoss', num_classes=1, dec_layers=1)
                ),
    auxiliary_head=dict(num_classes=150))

# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

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