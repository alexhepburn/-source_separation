"""Generate training data to be used in source_separation."""
import librosa
import numpy as np
import os
import h5py
import scipy
from optparse import OptionParser
import sys
from distutils.util import strtobool
from options import get_opt
import medleydb as mdb

def user_query(question):
    print('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(raw_input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')

class FeatureExtraction():
    def __init__(self, opt, instruments):
        self.instr = instruments
        self.dataset_dir = opt['d_path']
        self.n_fft = opt['n_fft']
        self.timesteps = opt['timesteps']
        self.features = opt['features']
        self.generate_dicts()
        if os.path.isfile('train_data.hdf5'):
            if user_query(('hdf5 data files already exist. Do you want to'
                          ' overwrite?')):
                            self.train_h5 = h5py.File('train_data.hdf5', 'w')
                            self.test_h5 = h5py.File('test_data.hdf5', 'w')
                            self.valid_h5 = h5py.File('valid_data.hdf5', 'w')
            else:
                sys.exit('No overwrite, exiting')
        else:
            self.train_h5 = h5py.File('train_data.hdf5', 'w')
            self.test_h5 = h5py.File('test_data.hdf5', 'w')
            self.valid_h5 = h5py.File('valid_data.hdf5', 'w')
        self.write_h5s()

    def generate_dicts(self):
        "Generate lists of folders that contain the data."
        instr_files = mdb.get_files_for_instrument(self.instr[0])
        instr_list = list(instr_files)[0:30]
        print 'Number of songs found with mixture and instruments defined:' \
              '{}'.format(len(instr_list))
        mix_list = []
        # Check it works with 20 files first
        for x in instr_list:
            base_file = os.path.dirname(os.path.dirname(x))
            for file in os.listdir(base_file):
                # Some hidden files start with ._ then mixture
                if 'MIX' in file and '._' not in file:
                    mix_list.append(base_file + '/' + file)
                    break
        # Randomly shuffle mix & instr
        comb = zip(mix_list, instr_list)
        np.random.shuffle(comb)
        mix_list[:], instr_list[:] = zip(*comb)

        self.train_dict = {
                           'mix': mix_list[0:int(len(mix_list)/2)],
                           'instr': instr_list[0:int(len(mix_list)/2)]
                           }
        self.test_dict = {
                           'mix': mix_list[int(len(mix_list)/2):
                                           int(3*len(mix_list)/4)],
                           'instr': instr_list[int(len(mix_list)/2):
                                               int(3*len(mix_list)/4)]
                           }
        self.valid_dict = {
                           'mix': mix_list[int(3*len(mix_list)/4):
                                           len(mix_list)],
                           'instr': instr_list[int(3*len(mix_list)/4):
                                               len(mix_list)]
                           }

    def write_file(self, h5_file, dict):
        mix = dict['mix']
        instr = dict['instr']
        mix_data_lst = []
        instr_data_lst = []
        num_samples = 0
        if len(mix) != len(instr):
            sys.exit('Error: mixture and instruments have different number'
                     'of elements.')
        for i in range(len(mix)):
            print 'Reading in ' + mix[i]
            S_m, sr_ = self.get_data(mix[i])
            S_i, sr_ = self.get_data(instr[i])

            conc_m = np.hstack((S_m.real, S_m.imag))
            conc_m = np.reshape(conc_m, (-1, self.timesteps, self.features))
            conc_i = np.hstack((S_i.real, S_i.imag))
            conc_i = np.reshape(conc_i, (-1, self.timesteps, self.features))
            num_samples += conc_m.shape[0]

            mix_data_lst.append(conc_m)
            instr_data_lst.append(conc_i)

        mix_out = self.lst_to_matrix(mix_data_lst, num_samples)
        instr_out = self.lst_to_matrix(instr_data_lst, num_samples)
        m_dset = h5_file.create_dataset("mixture", data=mix_out, chunks=True)
        i_dset = h5_file.create_dataset("instr", data=instr_out, chunks=True)
        h5_file['file_names'] = mix

    def lst_to_matrix(self, lst, num):
        out = np.empty((num, self.timesteps, self.features))
        start = 0
        end = 0
        for d in lst:
            end += d.shape[0]
            out[start:end, :, :] = d
            start += d.shape[0]
        return out

    def write_h5s(self):
        print 'Processing Training dataset...'
        self.write_file(self.train_h5, self.train_dict)
        print 'Processing  Testing dataset...'
        self.write_file(self.test_h5, self.test_dict)
        print 'Processing  Validation dataset...'
        self.write_file(self.valid_h5, self.valid_dict)
        self.train_h5.close()
        self.test_h5.close()
        self.valid_h5.close()

    def get_data(self, file):
        """Read in audio file and computes STFT."""
        y, sr_ = librosa.load(file, duration=120)
        if y.shape[0] < 2000000:
            y, sr_ = librosa.load(file, duration=30)
        # y_harm, y_perc = librosa.effects.hpss(y)
        S = librosa.core.stft(y=y, n_fft=self.n_fft).transpose()
        return S, sr_


def option_callback(option, opt, value, parse):
    setattr(parse.values, option.dest, value.split(','))

if __name__ == '__main__':
    parse = OptionParser()
    parse.add_option('--instruments', '-i', type='string', action='callback',
                     callback=option_callback, dest='instruments')
    (options, args) = parse.parse_args()
    if len(options.instruments) != 1:
        sys.exit('1 instrument must be defined using -i option.')
    opt = get_opt()
    f = FeatureExtraction(opt, options.instruments)
