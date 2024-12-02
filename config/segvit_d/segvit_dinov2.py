# model settings
_base_ = [
    '../_base_/datasets/ade20k_518.py',
    '../_base_/schedules/schedule_80k.py', '../_base_/default_runtime.py'
]

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (518, 518)
img_size = 518
in_channels = 768
out_indices = [5, 7, 11]
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='Dinov2',
        freeze=False,
        out_indices=out_indices,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/ckpt/dinov2_vitb14_pretrain.pth',
            prefix=None,
        )),
    decode_head=dict(
        type='ATMHeadV2',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        num_classes=150,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=in_channels // 2,
        loss_decode=dict(
            type='ATMLoss', num_classes=1, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(518, 518), stride=(341, 341)),)
# optimizer = dict(lr=0.001, weight_decay=0.0)
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
             window_size=50)
    ],
)

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
find_unused_parameters=True