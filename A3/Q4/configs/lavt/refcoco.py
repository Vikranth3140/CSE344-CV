# LAVT model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='LAVT',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        patch_size=4,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        type='LAVTHead',
        in_channels=[128, 256, 512, 1024],
        channels=256,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        num_classes=1,
    ),
    text_encoder=dict(
        type='BertModel',
        pretrained='bert-base-uncased',
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
