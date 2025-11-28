from collections import namedtuple

model_arg = namedtuple('model_arg', ('nf',  # number of feature channel
                                     'groups',  # number of deformable convolution group
                                     'upscale_factor',  # model upscale factor
                                     'format',  # model I/O format (rgb, yuv420)
                                     'layers',  # feature fusion pyramid layers
                                     'cycle_count'  # mutual cycle count
                                     ))

model_arg_mamba = namedtuple('model_arg', ('nf',  # number of feature channel
                                     'groups',  # number of deformable convolution group
                                     'img_size', # image size of input，for patch_embed
                                     'upscale_factor',  # model upscale factor
                                     'format',  # model I/O format (rgb, yuv420)
                                     'layers',  # feature fusion pyramid layers
                                     ))

model_arg_mamba_ff = namedtuple('model_arg', ('nf',  # number of feature channel
                                     'img_size', # image size of input，for patch_embed
                                     'upscale_factor',  # model upscale factor
                                     'format',  # model I/O format (rgb, yuv420)
                                     'layers',  # feature fusion pyramid layers
                                     ))
