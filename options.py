"""
Define option parameters for source_separation.

Also used by feature extraction.
"""


def get_opt():
    opt = {}
    opt['features'] = 1026
    opt['timesteps'] = 2
    opt['dropout'] = 0.25
    opt['layer_init'] = 'he_normal'
    opt['gamma'] = 100
    opt['epoch'] = 1000
    opt['batch_size'] = 128
    opt['plot'] = False
    opt['n_fft'] = 1024
    opt['pre_train_D'] = True
    opt['GRU_layers'] = 1
    opt['IS_MIR_EVAL'] = False

    # Learning rates
    opt['G_lr'] = 0.001
    opt['D_lr'] = 0.001
    opt['GAN_lr'] = 0.001
    return opt
