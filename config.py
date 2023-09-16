log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoderHV2',
    backbone=dict(
        type='mit_b2',
        style='pytorch',
    ),
    decode_head=dict(
        type='HV2Head',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(
        work_dir=''),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(mode='slide2', crop_size=(256, 256), stride=(256, 256), crop_output_size=(200, 200))
)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsMat', with_type=False),
    dict(type='Resize', ratio_range=(0.8, 1.2), multiscale_mode='range'),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomFlip', direction='vertical', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='Normalize99', add_noise=True),
    dict(type='GetHVMap'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_semantic_seg', 'gt_hv_map'
        ],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize99'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NucleiHV2Dataset',
        data_root='data/PanNuke/',
        img_dir='Images_fold_12/',
        ann_dir='Labels_fold_12/',
        img_suffix=".png",
        seg_map_suffix=".mat",
        pipeline=source_train_pipeline
    ),
    val=dict(
        type='NucleiHV2Dataset',
        data_root='data/PanNuke/',
        img_dir='Images_fold_3/',
        ann_dir='Labels_fold_3/',
        img_suffix=".png",
        seg_map_suffix=".mat",
        pipeline=test_pipeline),
    test=dict(
        type='NucleiHV2Dataset',
        data_root='data/',
        img_dir='',
        img_suffix=".tif",
        seg_map_suffix=".mat",
        pipeline=test_pipeline))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 42
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='all')
work_dir = './work_dirs/hv_head/pannuke12_aug_beat_stardist/'
