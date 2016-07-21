"""
Define option parameters for source_separation.

Also used by feature extraction.
"""


def get_opt():
    opt = {}
    opt['features'] = 1026
    opt['timesteps'] = 17
    opt['d_path'] = '/home/alexhepburn/Documents/Datasets/Separation dataset/'
    opt['dropout'] = 0.25
    opt['layer_init'] = 'glorot_uniform'
    opt['gamma'] = 0.5  # Parameter in front of similarity term for cost.
    opt['conv_masks'] = 3
    opt['epoch'] = 10
    opt['batch_size'] = 200
    opt['plot'] = False
    opt['n_fft'] = 1024
    return opt
