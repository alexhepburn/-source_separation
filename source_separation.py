"""GAN to separate instruments from a mixture."""
import numpy as np
from keras import backend as K
import theano
from keras.models import Model, Sequential
from keras.layers import Dense, TimeDistributed, Dropout, GRU, \
                         BatchNormalization
from keras.utils.visualize_util import plot
from keras.objectives import mean_squared_error
from keras.optimizers import Adagrad
import h5py
import mir_eval
from optparse import OptionParser
from options import get_opt
from scipy.io import wavfile
import time
import librosa
import cPickle as pickle


class Source_Separation_LSTM():
    """Class that separates instruments from a mixture using LSTM."""

    def __init__(self, options):
        """Initialise network structure."""
        # Pre-define batch for use in objective function feature_matching
        self.timesteps = options['timesteps']
        self.features = options['features']
        self.gamma = options['gamma']
        self.drop = options['dropout']
        self.plot = options['plot']
        self.epoch = options['epoch']
        self.batch_size = options['batch_size']
        self.init = options['layer_init']
        self.pre_train_D = options['pre_train_D']
        self.num_GRU = options["GRU_layers"]
        self.is_mir_eval = options["IS_MIR_EVAL"]
        self.G_lr = options["G_lr"]
        self.D_lr = options["D_lr"]
        self.GAN_lr = options["GAN_lr"]
        self.mir_eval = {'SAR': [],
                         'SIR': [],
                         'SDR': []}
        self.batch = np.zeros((self.batch_size, self.timesteps, self.features))
        self.D__init__()
        self.G__init__()
        self.GAN__init__()
        if self.plot:
            plot(self.GAN, to_file='model.png')

    def G__init__(self):
        self.G = Sequential()
        self.G.add(TimeDistributed(Dense(self.features, init=self.init), input_shape=(self.timesteps, self.features)))
        for layer in range(self.num_GRU):
            self.G.add(GRU(513, return_sequences=True))
        self.G.add(TimeDistributed(Dense(self.features, init=self.init)))
        self.G.compile(loss=self.objective, optimizer=Adagrad(lr=self.G_lr,
                                                              epsilon=1e-08))

    def D__init__(self):
        self.D = Sequential()
        self.D.add(TimeDistributed(Dense(self.features, activation='relu',
                                         init=self.init),
                                   input_shape=(self.timesteps,
                                                self.features)))
        self.D.add(Dropout(self.drop))
        self.D.add(TimeDistributed(Dense(513, activation='relu',
                                         init=self.init)))
        self.D.add(TimeDistributed(Dense(1, activation='sigmoid',
                                         init=self.init)))
        self.D.compile(loss='binary_crossentropy',
                       optimizer=Adagrad(lr=self.D_lr, epsilon=1e-08),
                       metrics=["accuracy"])

    def GAN__init__(self):
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.D.trainable = False
        self.GAN.add(self.D)
        self.GAN.compile(loss='binary_crossentropy',
                         optimizer=Adagrad(lr=self.GAN_lr, epsilon=1e-08))

    def objective(self, out_true, out_pred):
        mse = mean_squared_error(out_true, out_pred)
        return mse

    # not quite working
    def feature_matching(self, out_true, out_pred):
        "Feature matching objective function for use in G."
        mse = mean_squared_error(out_true, out_pred)
        activations = K.function([self.D.layers[0].input, K.learning_phase()],
                                  [self.D.layers[1].output,])
        # Inputs to theano functions must be numeric not tensors like out_true
        pred_batch = self.D.predict(self.batch, batch_size=self.batch_size)
        pred_activ = activations([pred_batch, 0])[0]
        true_activ = activations([self.batch, 0])[0]
        feat_match = mean_squared_error(true_activ, pred_activ)
        return mse + self.gamma * feat_match

    def fit(self, input, out1_true, valid_in, valid_out):
        """Train neural network given input data and corresponding output."""
        start_time = time.time()
        if self.pre_train_D:
            self.pre_trainD(input[0:2000], out1_true[0:2000])
        d_loss = [2, 0.5]
        self.g_loss_valid = []
        self.g_loss_valid.append(float('Inf'))
        for epoch in range(self.epoch):
            self.epoch = epoch
            print "Epoch: {}".format(epoch)
            p = np.random.permutation(input.shape[0]-1)
            start, end = 0, 0
            while len(p) - end > self.batch_size:
                end += self.batch_size
                ind = p[start:end]
                self.batch = input[ind, :, :]
                batch_out = out1_true[ind, :, :]
                self.train_on_batch(self.batch, batch_out)
                start += self.batch_size
            G_val_loss = self.G.evaluate(valid_in, valid_out,
                                         batch_size=self.batch_size)
            d_label = np.ones((valid_in.shape[0], self.timesteps, 1))
            GAN_val_loss = self.GAN.evaluate(valid_in, d_label, self.batch_size)
            valid_pred = self.G.predict(valid_in, batch_size=self.batch_size)
            valid_loss = G_val_loss
            if (min(self.g_loss_valid) > valid_loss):
                print "Saving weights, validation loss improved to {}".format(valid_loss)
                self.G.save_weights("bestweights.hdf5", overwrite=True)
            else:
                print "Validation loss: {}".format(valid_loss)
            self.G.save_weights("weights.hdf5", overwrite=True)
            self.g_loss_valid.append(valid_loss)
            self.update_mir_eval(valid_out, valid_pred)
            np.save('loss_values.npy', self.g_loss_valid)
            print "Elapsed time: {}".format(time.time()-start_time)

    def train_on_batch(self, batch_in, batch_out):
        y = np.concatenate((np.ones((self.batch_size, self.timesteps, 1)),
                            np.zeros((self.batch_size, self.timesteps, 1))))
        self.pred = self.G.predict(self.batch)
        g_loss = self.G.train_on_batch(self.batch, batch_out)
        D_in = np.concatenate((batch_out, self.pred))
        d_loss = self.D.train_on_batch(D_in, y)
        y_train = np.ones((self.batch_size, self.timesteps, 1))
        GAN_loss = self.GAN.train_on_batch(self.batch, y_train)
        print "d_loss: {}, g_loss: {}, GAN_loss: {}".format(d_loss[1],
                                                            g_loss,
                                                            GAN_loss)

    def update_mir_eval(self, true_out, pred_out):
        if self.epoch % 10 == 0 or self.is_mir_eval:
            pred = self.time_signal(pred_out)
        if self.epoch % 10 == 0:
            stri = 'Output/Epoch' + str(self.epoch) + '.wav'
            wavfile.write(stri, 22050, pred)
        if self.is_mir_eval:
            out = self.time_signal(true_out)
            (sdr, sir, sar, _) = mir_eval.separation.bss_eval_sources(out, pred)
            print "SDR: {}, SIR: {}, SAR: {}".format(sdr, sir, sar)
            self.mir_eval['SDR'].append(sdr)
            self.mir_eval['SIR'].append(sir)
            self.mir_eval['SAR'].append(sar)
            with open('MIR_eval_values.pickle', 'wb') as handle:
                pickle.dump(self.mir_eval, handle)

    def pre_trainD(self, input, output):
        self.D.trainable = True
        n_samples = input.shape[0]
        pred = self.G.predict(input)
        y = np.concatenate((np.ones((n_samples, self.timesteps, 1)),
                            np.zeros((n_samples, self.timesteps, 1))))
        x = np.concatenate((output, pred))
        self.D.fit(x, y, batch_size=self.batch_size, nb_epoch=1)
        self.D.trainable = False


    def predict(self, test, batch_size):
        """Predict output given input using G."""
        out1 = self.G.predict(test, batch_size)
        return out1

    def load_weights(self, path):
        """Load weights from saved weights file in hdf5."""
        self.GAN.load_weights(path)

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

    def time_signal(self, input):
        "Turn input or output into time signal."
        x = np.reshape(input, (-1, self.features)).transpose()
        x = self.conc_to_complex(x)
        return librosa.core.istft(x)

if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option('--load', '-l', action='store_true', dest='load',
                     default=False, help='Loads weights from weights.')
    (options, args) = parse.parse_args()
    print 'Initialising model'
    model = Source_Separation_LSTM(get_opt())
    v_mixture, v_instr = model.open_h5('valid_data.hdf5')
    if options.load is False:
        print 'Training model'
        train_mixture, train_instr = model.open_h5('train_data.hdf5')
        wavfile.write('train_x.wav', 22050, model.time_signal(train_mixture))
        wavfile.write('train_y.wav', 22050, model.time_signal(train_instr))
        model.fit(train_mixture, train_instr, v_mixture, v_instr)
    else:
        print 'Loading weights from weights.hdf5'
        model.G.load_weights('weights.hdf5')

    print "Predicting on validation data"
    out = model.predict(v_mixture, batch_size=128)
    wavfile.write('test_out.wav', 22050, model.time_signal(out))
    wavfile.write('mix.wav', 22050, model.time_signal(v_mixture-out))
    wavfile.write('test.wav', 22050, model.time_signal(v_instr))
