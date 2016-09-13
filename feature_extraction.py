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
import math
import sklearn as sk

def user_query(question):
    print('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(raw_input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')

class FeatureExtraction():
    def __init__(self, opt, instruments):
        print(mdb.get_valid_instrument_labels())
        self.instr = instruments
        self.n_fft = opt['n_fft']
        self.timesteps = opt['timesteps']
        self.features = opt['features']
        self.mix_scaler = sk.preprocessing.StandardScaler()
        self.instr_scaler = sk.preprocessing.StandardScaler()
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
        tempinstr = list(instr_files)[0:30]
        instr_list = []
        for song in tempinstr:
            if 'Rockabilly' not in song:
                instr_list.append(song)
        print 'Number of songs found with mixture and instruments defined:' \
              '{}'.format(len(instr_list))
        mix_list = []
        for x in instr_list:
            base_file = os.path.dirname(os.path.dirname(x))
            for file in os.listdir(base_file):
                # Some files start with ._ then mixture
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

    def write_file(self, h5_file, dict, mode):
        mix = dict['mix']
        instr = dict['instr']
        mix_data_lst = []
        instr_data_lst = []
        num_samples = 0
        if len(mix) != len(instr):
            sys.exit('Error: mixture and instruments have different number'
                     'of elements.')
        for i in range(len(mix)):
            print 'Reading in ' + instr[i]
            S_m, sr_ = self.get_data(mix[i])
            S_i, sr_ = self.get_data(instr[i])

            # numsamples is the most samples of length self.timesteps that
            # sample can have
            numsamples = math.trunc(S_m.shape[0] / self.timesteps)
            if numsamples != 0:
                S_m = S_m[0:numsamples*self.timesteps, :]
                S_i = S_i[0:numsamples*self.timesteps, :]
                conc_m = np.hstack((S_m.real, S_m.imag))
                conc_m = np.reshape(conc_m, (-1, self.timesteps,
                                             self.features))
                conc_i = np.hstack((S_i.real, S_i.imag))
                print conc_i.shape
                conc_i = np.reshape(conc_i, (-1, self.timesteps,
                                             self.features))
                num_samples += conc_m.shape[0]

                mix_data_lst.append(conc_m)
                instr_data_lst.append(conc_i)

        mix_out = self.lst_to_matrix(mix_data_lst, num_samples)
        instr_out = self.lst_to_matrix(instr_data_lst, num_samples)
        if mode == 'train' or 'valid':
            del_ind = []
            for i in range(instr_out.shape[0]):
                if np.all(instr_out[i, :, :] == 0):
                    del_ind.append(i)
            print "Deleting {} empty samples from {}".format(len(del_ind), mode)
            instr_out = np.delete(instr_out, del_ind, axis=0)
            mix_out = np.delete(mix_out, del_ind, axis=0)
        #if mode == 'train':
        #    print 'Fitting StandardScaler'
        #    for i in range(mix_out.shape[0]):
        #        self.mix_scaler.partial_fit(mix_out[i, :, :])
        #        self.instr_scaler.partial_fit(instr_out[i, :, :])
        #for i in range(mix_out.shape[0]):
        #    mix_out[i, :, :] = self.mix_scaler.transform(mix_out[i, :, :])
        #    instr_out[i, :, :] = self.instr_scaler.transform(instr_out[i, :, :])
        print "{} samples".format(instr_out.shape[0])
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
        self.write_file(self.train_h5, self.train_dict, 'train')
        print 'Processing  Testing dataset...'
        self.write_file(self.test_h5, self.test_dict, 'test')
        print 'Processing  Validation dataset...'
        self.write_file(self.valid_h5, self.valid_dict, 'valid')
        self.train_h5.close()
        self.test_h5.close()
        self.valid_h5.close()

    def get_data(self, file):
        """Read in audio file and computes STFT."""
        y, sr_ = librosa.load(file)
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
