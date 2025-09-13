import ml_collections

def default():
    config = ml_collections.ConfigDict()
    config.image_height = 480
    config.image_width = 640
    config.in_channels = 3
    config.out_channels = [32, 64, 128, 256]
    config.rep = [1, 2, 2, 1]
    config.stride = (8, 16, 32)
    config.num_classes = 80
    config.num_masks = 32
    return config

def config_N():
    config = default()
    return config

def config_S():
    config = default()
    config.out_channels = [64, 128, 256, 512]
    return config

def config_M():
    config = default()
    config.out_channels = [96, 192, 384, 576]
    config.rep = [2, 4, 4, 2]
    return config

def config_L():
    config = default()
    config.out_channels = [128, 256, 512, 512]
    config.rep = [3, 6, 6, 3]
    return config

def config_X():
    config = default()
    config.out_channels = [160, 320, 640, 640]
    config.rep = [3, 6, 6, 3]
    return config

YoloV8I_CONFIGS = {
    "n": config_N(),
    "s": config_S(),
    "m": config_M(),
    "l": config_L(),
    "x": config_X()
}