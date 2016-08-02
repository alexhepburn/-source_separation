"""GAN to separate instruments from a mixture."""
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, TimeDistributed, LSTM, Input, Dropout, \
                         Convolution1D, Flatten, GRU
from keras.utils.visualize_util import plot
from keras.layers.advanced_activations import LeakyReLU
from keras.objectives import mean_squared_error
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
import h5py
import librosa
from optparse import OptionParser
from options import get_opt
from scipy import io
import time

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

    # add validation for early stopping
    def fit(self, input, out1_true, valid_in, valid_out):
        """Train neural network given input data and corresponding output."""
        start_time = time.time()
        y = np.concatenate((np.ones((self.batch_size, self.timesteps, 1)),
                            np.zeros((self.batch_size, self.timesteps, 1))))
        if self.pre_train_D:
            ind = np.random.permutation(400)
            predi = self.G.predict(input[0:400])
            y2 = np.concatenate((np.ones((400, self.timesteps, 1)),
                                np.zeros((400, self.timesteps, 1))))
            d2 = np.concatenate((input[0:400], predi))
            self.D.fit(d2, y2, batch_size=self.batch_size, nb_epoch=5)
        self.g_loss_valid = []
        self.g_loss_valid.append(float('Inf'))
        for epoch in range(self.epoch):
            print "Epoch: {}".format(epoch)
            p = np.random.permutation(input.shape[0]-1)
            start = 0
            end = 0

            while len(p) - end > self.batch_size:
                end += self.batch_size
                ind = p[start:end]
                self.batch = input[ind, :, :]
                batch_out = out1_true[ind, :, :]
                g_loss = self.G.train_on_batch(self.batch, batch_out)
                #self.pred = self.G.predict(self.batch)
                #D_in = np.concatenate((self.batch, self.pred))
                #self.D.trainable = True
                #d_loss = self.D.train_on_batch(D_in, y)
                #self.D.trainable = False
                #self.GAN.train_on_batch(self.batch, np.zeros((self.batch_size, self.timesteps, 1)))
                print "d_loss: {}, g_loss: {}".format(0, g_loss)
                start += self.batch_size
            valid_loss = (self.G.evaluate(valid_in, valid_out, batch_size=64))
            if (min(self.g_loss_valid) > valid_loss):
                print "Saving weights, validation loss improved to {}".format(valid_loss)
                self.GAN.save_weights("weights.hdf5", overwrite=True)
            self.g_loss_valid.append(valid_loss)
            print "Elapsed time: {}".format(time.time()-start_time)
        np.save("Validation_losses", self.g_loss_valid)

    def predict(self, test, batch_size):
        """Predict output given input using neural network."""
        out1 = self.G.predict(test, batch_size)
        return out1

    def load_weights(self, path):
        """Load weights from saved weights file in hdf5."""
        self.GAN.load_weights(path)

    def __init__(self, options):
        """Initialise network structure."""
        self.timesteps = options['timesteps']
        self.features = options['features']
        self.out1_true = 0
        self.out2_true = 0
        self.gamma = options['gamma']
        self.drop = options['dropout']
        self.plot = options['plot']
        self.epoch = options['epoch']
        self.batch_size = options['batch_size']
        self.init = options['layer_init']
        self.pre_train_D = options['pre_train_D']
        self.G__init__()
        self.D__init__()
        self.GAN__init__()
        if self.plot:
            plot(self.GAN, to_file='model.png')
        # Save best weights to hdf5 file
        self.checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1,
                                            save_best_only=True)

    def G__init__(self):
        mix = Input(shape=(self.timesteps, self.features), dtype='float32')
        # GRU's yield similar performance but are more efficient than LSTM
        l1 = GRU(self.features, return_sequences=True, activation='relu',
                 init=self.init)(mix)
        l1 = Dropout(self.drop)(l1)
        l2 = GRU(2000, return_sequences=True, activation='relu',
                 init=self.init)(l1)
        l2 = Dropout(self.drop)(l2)
        l3 = GRU(self.features, return_sequences=True, activation='relu',
                 init=self.init)(l2)
        l3 = Dropout(self.drop)(l3)
        d = TimeDistributed(Dense(self.features, init=self.init))(l2)
        self.G = Model(input=mix, output=d)
        self.G.compile(loss='mse', optimizer=Adagrad(lr=0.0001, epsilon=1e-08))

    def D__init__(self):
        inp = Input(shape=(self.timesteps, self.features), dtype='float32')
        # without dropout D 50% -> 66% from pretraining
        d_1 = TimeDistributed(Dense(50, activation='relu', init=self.init))(inp)
        #d_1 = Dropout(self.drop)(d_1)
        d_2 = TimeDistributed(Dense(50, activation='relu', init=self.init))(d_1)
        #d_2 = Dropout(self.drop)(d_2)
        d_3 = TimeDistributed(Dense(50, activation='relu', init=self.init))(d_2)
        #d_3 = Dropout(self.drop)(d_3)
        d_v = TimeDistributed(Dense(1, activation='relu',
                                    init=self.init))(d_3)
        self.D = Model(inp, d_v)
        self.D.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.001, epsilon=1e-08),
                       metrics=["accuracy"])

    def GAN__init__(self):
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.D.trainable = False
        self.GAN.add(self.D)
        self.GAN.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.001, epsilon=1e-08))

    def open_h5(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            mix = np.array(f.get('mixture'))
            instr = np.array(f.get('instr'))
        return mix, instr

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
    v_mixture, v_instr1 = model.open_h5('valid_data.hdf5')
    if options.load is False:
        print 'Training model'
        train_mixture, train_instr1 = model.open_h5('train_data.hdf5')
        model.fit(train_mixture, train_instr1, v_mixture, v_instr1)
    else:
        print 'Loading weights from weights.hdf5'
        model.load_weights('weights.hdf5')

    print "Predicting on validation data"
    out1 = model.predict(v_mixture, batch_size=64)

    testinstr1 = np.reshape(v_instr1, (v_instr1.shape[0]*model.timesteps, model.features)).transpose()
    mix = np.reshape(v_mixture, (v_mixture.shape[0]*model.timesteps, model.features)).transpose()
    out1 = np.reshape(out1, (out1.shape[0]*model.timesteps, model.features)).transpose()
    mixcomp = model.conc_to_complex(mix)
    test1comp = model.conc_to_complex(testinstr1)
    out1_comp = model.conc_to_complex(out1)

    instr1_softmask = librosa.util.softmask(np.absolute(test1comp), np.absolute(mixcomp), power=2)
    out1_comp = out1_comp * instr1_softmask

    mp31 = librosa.core.istft(out1_comp)
    intr1 = librosa.core.istft(test1comp)
    mix = librosa.core.istft(mixcomp)

    io.wavfile.write('test_out1.wav', 22050, mp31)
    io.wavfile.write('mix.wav', 22050, mix)
    io.wavfile.write('test1.wav', 22050, intr1)
