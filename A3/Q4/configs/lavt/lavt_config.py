# LAVT configuration based on the README

# Do not use mmcv - using pure Python dictionaries instead
# Model configuration settings from the LAVT README
lavt_config = {
    # Swin Transformer backbone settings
    'backbone': {
        'type': 'swin',
        'pretrain_img_size': 384,
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': 12,
        'drop_path_rate': 0.3,
        'patch_size': 4,
        'mlp_ratio': 4,
        'strides': (4, 2, 2, 2),
        'pretrained_weights': './pretrained_weights/swin_base_patch4_window12_384_22k.pth'
    },

    # Text encoder settings
    'text_encoder': {
        'type': 'bert',
        'pretrained': 'bert-base-uncased',
        'trainable': True,
        'version': '3.0.2'
    },

    # Decode head settings
    'decode_head': {
        'in_channels': [128, 256, 512, 1024],
        'channels': 256,
        'dropout_ratio': 0.1,
        'num_classes': 1
    },

    # Training settings from README
    'train': {
        'img_size': 480,
        'batch_size': 8,
        'lr': 0.00005,
        'weight_decay': 1e-2,
        'epochs': 40,
        'workers_per_gpu': 4
    }
}
