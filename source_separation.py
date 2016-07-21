"""Using LSTM layers to separate audio from TRIOS_dataset."""
import numpy as np
from keras.models import Model
from keras.layers import Dense, TimeDistributed, LSTM, Input, Dropout, \
                         Convolution1D, Flatten
from keras.utils.visualize_util import plot
from keras.objectives import mean_squared_error
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
import h5py
import librosa
from optparse import OptionParser
from options import get_opt
from scipy import io
import sys


class Source_Separation_LSTM():
    """Class that separates instruments from a mixture using LSTM."""

    def objective_1(self, out1_true, out1_pred):
        """First objective function.

        Using mean squared error between the true and predicted values & the
        difference between predicted out1 and true out2 to make out1 and out2
        as different as possible.
        """
        mse = mean_squared_error(out1_true, out1_pred)
        diff = mean_squared_error(out1_pred, self.out2_true)
        return mse - self.gamma * diff

    def objective_2(self, out2_true, out2_pred):
        """Second objective function.

        Using mean squared error between the true and predicted values & the
        difference between predicted out2 and true out1 to make out1 and out2
        as different as possible.
        """
        mse = mean_squared_error(out2_true, out2_pred)
        diff = mean_squared_error(out2_pred, self.out1_true)
        return mse - self.gamma * diff

    def fit(self, input, out1_true, out2_true, valid_in, valid_out):
        """Train neural network given input data and corresponding output."""
        self.out1_true = out1_true
        self.out2_true = out2_true
        self.model.fit(input, [out1_true, out2_true], nb_epoch=self.epoch,
                       batch_size=self.batch_size,
                       callbacks=[self.checkpointer],
                       validation_data=(valid_in, valid_out))

    def predict(self, test, batch_size):
        """Predict output given input using neural network."""
        out1, out2 = self.model.predict(test, batch_size)
        return out1, out2

    def load_weights(self, path):
        """Load weights from saved weights file in hdf5."""
        self.model.load_weights(path)

    def make_train(self, net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def __init__(self, options):
        """Initialise network structure."""
        self.timesteps = options['timesteps']
        self.features = options['features']
        self.out1_true = 0
        self.out2_true = 0
        self.gamma = options['gamma']
        self.drop = options['dropout']
        self.conv_masks = options['conv_masks']
        self.plot = options['plot']
        self.epoch = options['epoch']
        self.batch_size = options['batch_size']
        self.init = options['layer_init']
        self.S__init__()
        self.D__init__()
        make_trainable(self.D, False)
        gan_in = Input(batch_size=(None, self.timesteps, self.features),
                       dtype='float32')
        H = self.S(gan_in)
        gan_V = self.D(H)
        self.GAN = Model(gan_in, gan_V)
        GAN.complile(loss='categorical_crossentropy', optimizerrrr='Adagrad')
        if self.plot:
            plot(self.S, to_file='model.png')
        # Save best weights to hdf5 file
        self.checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1,
                                            save_best_only=True)

    def S__init__(self):
        mix = Input(batch_shape=(None, self.timesteps, self.features),
                    dtype='float32')
        conv = Convolution1D(self.features, self.conv_masks,
                             border_mode='same', init=self.init)(mix)
        lstm = LSTM(self.features, return_sequences=True, init=self.init)(conv)
        lstm2 = LSTM(self.features, return_sequences=True, init=self.init)(lstm)
        lstm2_drop = Dropout(self.drop)(lstm2)
        self.S_out1 = TimeDistributed(Dense(self.features,
                                            activation='relu',
                                            init=self.init))(lstm2_drop)
        self.S_out2 = TimeDistributed(Dense(self.features,
                                            activation='relu',
                                            init=self.init))(lstm2_drop)

        self.S = Model(input=[self.mix], output=[self.out1, self.out2])
        self.S.compile(loss=[self.objective_1, self.objective_2],
                       optimizer='Adagrad')

    # Could pretrain D for faster convergence
    def D__init__(self):
        inp = Input(batch_shape=(None, self.timesteps, self.features),
                    dtype='float32')
        d_1 = Dense(self.features, activation='relu', init=self.init)(inp)
        d_2 = Dense(100, activation='relu', init_self.init)(d_1)
        d_3 = Dropout(self.dropout)(d_2)
        d_4 = Flatten()(d_3)
        d_v = Dense(2, activation='softmax')(d_4)
        self.D = Model.(inp, d_v)
        self.D.compile(loss='categorical_crossentropy', optimizer='Adagrad')

    def h5_to_matrix(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            song_names = f.keys()[:]
            del song_names[song_names.index('count')]
            num = np.array(f.get('count'))
            instr_names = f[song_names[0]].keys()
            del instr_names[instr_names.index('mix')]
            mixture = np.zeros((num, self.timesteps, self.features),
                               dtype=np.float32)
            instr1 = np.zeros((num, self.timesteps, self.features),
                              dtype=np.float32)
            instr2 = np.zeros((num, self.timesteps, self.features),
                              dtype=np.float32)
            start = 0
            for song in song_names:
                if isinstance(f[song], h5py.Group) is True:
                    end = start + np.array(f[song]['mix']).shape[0]
                    mixture[start:end, :, :] = np.array(f[song]['mix'])
                    print 'Instrument 1: {}'.format(instr_names[0])
                    print 'Instrument 2: {}'.format(instr_names[1])
                    instr1[start:end, :, :] = np.array(f[song][instr_names[0]])
                    instr2[start:end, :, :] = np.array(f[song][instr_names[1]])
                    start = end
        return mixture, instr1, instr2


    def conc_to_complex(self, matrix):
        """Turn matrix in form [real, complex] to compelx number."""
        split = self.features/2
        end = matrix.shape[0]
        real = matrix[0:split, :]
        im = matrix[split:end, :]
        out = real + im * 1j
        return out

if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option('--load', '-l', action='store_true', dest='load',
                     default=False, help='Loads weights from weights.')
    (options, args) = parse.parse_args()
    print 'Initialising model'
    model = Source_Separation_LSTM(get_opt())
    v_mixture, v_instr1, v_instr2 = model.h5_to_matrix('valid_data.hdf5')
    if options.load is False:
        print 'Training model'
        train_mixture, train_instr1, train_instr2 = model.h5_to_matrix('train_data.hdf5')
        print 'Fitting model'
        model.fit(train_mixture, train_instr1, train_instr2,
                  valid_in=v_mixture, valid_out=[v_instr1, v_instr2])
    else:
        print 'Loading weights from weights.hdf5'
        model.load_weights('weights.hdf5')

    print "Predicting on validation data"
    [out1, out2] = model.predict(v_mixture, batch_size=100)
    out1 = np.reshape(out1, (out1.shape[0]*model.timesteps, model.features)).transpose()
    out2 = np.reshape(out2, (out2.shape[0]*model.timesteps, model.features)).transpose()
    testinstr1 = np.reshape(v_instr1, (v_instr1.shape[0]*model.timesteps, model.features)).transpose()
    testinstr2 = np.reshape(v_instr2, (v_instr2.shape[0]*model.timesteps, model.features)).transpose()
    mix = np.reshape(v_mixture, (v_mixture.shape[0]*model.timesteps, model.features)).transpose()
    out1_comp = model.conc_to_complex(out1)
    out2_comp = model.conc_to_complex(out2)
    mixcomp = model.conc_to_complex(mix)
    test1comp = model.conc_to_complex(testinstr1)
    test2comp = model.conc_to_complex(testinstr2)

    intr1 = librosa.core.istft(test1comp)
    intr2 = librosa.core.istft(test2comp)
    mp31 = librosa.core.istft(out1_comp)
    mp32 = librosa.core.istft(out2_comp)
    mix = librosa.core.istft(mixcomp)
    io.wavfile.write('mix.wav', 22050, mix)
    io.wavfile.write('test_out1.wav', 22050, mp31)
    io.wavfile.write('test_out2.wav', 22050, mp32)
    io.wavfile.write('test1.wav', 22050, intr1)
    io.wavfile.write('test2.wav', 22050, intr2)
