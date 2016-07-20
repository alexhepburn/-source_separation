"""Generate training data to be used in source_separation."""
import librosa
import numpy as np
import os
import h5py
import random
from optparse import OptionParser
import sys
from distutils.util import strtobool

timesteps = 17
features = 513


def user_query(question):
    print('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(raw_input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')

class FeatureExtraction():
    def __init__(self, dataset_dir, instruments):
        self.instr = instruments
        self.dataset_dir = dataset_dir
        self.n_fft = 1024
        self.song_dir = []
        for x in os.listdir(dataset_dir):
            xdir = os.listdir(dataset_dir + '/' + x)
            # check if song has a file that contains mix and the
            # 2 instruments that the features will be calculated
            # for
            if (any('mix' in word for word in xdir) and
                    any(self.instr[0] in word for word in xdir) and
                    any(self.instr[1] in word for word in xdir)):
                self.song_dir.append(x)
        random.shuffle(self.song_dir)
        print 'Number of songs found with mixture and instruments defined:' \
              '{}'.format(len(self.song_dir))
        # First half of list is training data
        self.train_list = self.song_dir[0:int(len(self.song_dir)/2)]
        # The next quater of list is test data
        self.test_list = self.song_dir[int(len(self.song_dir)/2 + 1):
                                       int(3*len(self.song_dir)/4)]
        # The last quater of list is validation data
        self.valid_list = self.song_dir[int(3*len(self.song_dir)/4 + 1):
                                        len(self.song_dir) - 1]
        if os.path.isfile('train_data.hdf5'):
            if user_query(('Train hdf5 file already exists. Do you want to'
                          ' overwrite?')):
                            self.train_h5 = h5py.File('train_data.hdf5', 'w')
                            self.test_h5 = h5py.File('test_data.hdf5', 'w')
                            self.valid_h5 = h5py.File('valid_data.hdf5', 'w')
            else:
                sys.exit('No overwrite')
        self.train_num = 0
        self.test_num = 0
        self.valid_num = 0
        self.write_h5s()

    def load_file(self, h5_file, list):
        num = 0
        for fold in list:
            list_ = os.listdir(self.dataset_dir + fold)
            print 'Reading in ' + fold
            # Store features for each instrument in the songs subgroup
            grp = h5_file.create_group(fold)
            mix, instr1, instr2 = False, False, False
            for file in list_:
                file_path = self.dataset_dir + fold + '/' + file
                if 'mix' in file and mix is False:
                    mix = True
                    S, sr_ = self.get_data(file_path)
                    S = np.reshape(S, (-1, timesteps, features))
                    num += S.shape[0]
                    grp['mix'] = S
                if self.instr[0] in file and instr1 is False:
                    print file
                    instr1 = True
                    S, sr_ = self.get_data(file_path)
                    S = np.reshape(S, (-1, timesteps, features))
                    num += S.shape[0]
                    grp[self.instr[0]] = S
                if self.instr[1] in file and instr2 is False:
                    print file
                    instr2 = True
                    S, sr_ = self.get_data(file_path)
                    S = np.reshape(S, (-1, timesteps, features))
                    num += S.shape[0]
                    grp[self.instr[1]] = S
        return num

    def write_h5s(self):
        print 'Processing Training dataset...'
        self.train_num = self.load_file(self.train_h5, self.train_list)
        print 'Processing  Testing dataset...'
        self.test_num = self.load_file(self.test_h5, self.test_list)
        print 'Processing  Validation dataset...'
        self.valid_num = self.load_file(self.valid_h5, self.valid_list)
        self.train_h5['count'] = self.train_num
        self.test_h5['count'] = self.test_num
        self.valid_h5['count'] = self.valid_num
        self.train_h5.close()
        self.test_h5.close()
        self.valid_h5.close()

    def get_data(self, file):
        """Read in data from all .wav files inside folder & computes STFT."""
        y, sr_ = librosa.load(file, duration=30)
        S = librosa.core.stft(y=y, n_fft=self.n_fft).transpose()
        return S, sr_

def option_callback(option, opt, value, parse):
    setattr(parse.values, option.dest, value.split(','))

if __name__ == '__main__':
    parse = OptionParser()
    parse.add_option('--instruments', '-i', type='string', action='callback',
                     callback=option_callback, dest='instruments')
    (options, args) = parse.parse_args()
    if len(options.instruments) != 2:
        sys.exit('2 instruments must be defined using -i option.')
    path = '/home/alexhepburn/Documents/Datasets/Separation dataset/'
    f = FeatureExtraction(path, options.instruments)
