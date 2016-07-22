"""
Define option parameters for source_separation.

Also used by feature extraction.
"""


def get_opt():
    opt = {}
    opt['features'] = 1026
    opt['timesteps'] = 8
    opt['d_path'] = '/home/alexhepburn/Documents/Datasets/Separation dataset/'
    opt['dropout'] = 0.25
    opt['gamma'] = 0.5  # Parameter in front of similarity term for cost.
    opt['conv_masks'] = 3
    opt['epoch'] = 20
    opt['batch_size'] = 600
    opt['plot'] = False
    opt['n_fft'] = 1024
    return opt
