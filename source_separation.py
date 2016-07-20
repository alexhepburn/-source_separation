"""Using LSTM layers to separate audio from TRIOS_dataset."""
import numpy as np
from keras.models import Model
from keras.layers import Dense, TimeDistributed, LSTM, Input, Dropout, \
                         Convolution1D
from keras.utils.visualize_util import plot
from keras.objectives import mean_squared_error
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
import h5py
import librosa
from optparse import OptionParser

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

    def fit(self, input, out1_true, out2_true, epoch, batch_size_, valid_in,
            valid_out):
        """Train neural network given input data and corresponding output."""
        self.out1_true = out1_true
        self.out2_true = out2_true
        self.model.fit(input, [out1_true, out2_true], nb_epoch=epoch,
                       batch_size=batch_size_, callbacks=[self.checkpointer],
                       validation_data=(valid_in, valid_out))

    def predict(self, test, batch_size):
        """Predict output given input using neural network."""
        out1, out2 = self.model.predict(test, batch_size)
        return out1, out2

    def load_weights(self, path):
        """Load weights from saved weights file in hdf5."""
        self.model.load_weights(path)

    def __init__(self, timesteps, features):
        """Initialise network structure."""
        self.timesteps = timesteps
        self.features = features
        self.out1_true = 0
        self.out2_true = 0
        self.gamma = 0.5
        mix = Input(batch_shape=(None, timesteps, features))
        self.conv = Convolution1D(513, 3, border_mode='same')(mix)
        self.lstm = LSTM(features, return_sequences=True)(self.conv)
        self.lstm2 = LSTM(features, return_sequences=True)(self.lstm)
        self.lstm2_drop = Dropout(0.25)(self.lstm2)
        self.out1 = TimeDistributed(Dense(features,
                                          activation='relu'))(self.lstm2_drop)
        self.out2 = TimeDistributed(Dense(features,
                                          activation='relu'))(self.lstm2_drop)

        self.model = Model(input=[mix], output=[self.out1, self.out2])
        self.model.compile(loss=[self.objective_1, self.objective_2],
                           optimizer='rmsprop')
        # plot(self.model, to_file='model.png')
        # Save best weights to hdf5 file
        self.checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1,
                                            save_best_only=True)


def h5_to_matrix(h5_file):
    with h5py.File(h5_file, 'r') as f:
        song_names = f.keys()[:]
        num = np.array(f.get('count'))
        instr_names = f[song_names[0]].keys()
        del instr_names[instr_names.index('mix')]
        mixture = np.zeros((num, 17, 513), dtype=np.complex64)
        instr1 = np.zeros((num, 17, 513), dtype=np.complex64)
        instr2 = np.zeros((num, 17, 513), dtype=np.complex64)
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

if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option('--load', '-l', action='store_true', dest='load',
                     default=False, help='Loads weights from weights.')
    (options, args) = parse.parse_args()
    print 'Initialising model'
    model = Source_Separation_LSTM(17, 513)
    if options.load is False:
        print 'Training model'
        print 'Reading in test data'
        train_mixture, train_instr1, train_instr2 = h5_to_matrix('train_data.hdf5')
        v_mixture, v_instr1, v_instr2 = h5_to_matrix('valid_data.hdf5')
        print 'Fitting model'
        model.fit(train_mixture, train_instr1, train_instr2, epoch=3,
                  batch_size_=200, valid_in=v_mixture,
                  valid_out=[v_instr1, v_instr2])
    else:
        print 'Loading weights from weights.hdf5'
        model.load_weights('weights.hdf5')
    test_mixture, test_instr1, test_instr2 = h5_to_matrix('test_data.hdf5')
    test_mixture = test_mixture[600:900, :, :]
    test_instr1 = test_instr1[600:900, :, :]
    test_instr2 = test_instr2[600:900, :, :]
    [out1, out2] = model.predict(test_mixture, batch_size=1)
    out1 = np.reshape(out1, (300*17, 513)).transpose()
    out2 = np.reshape(out2, (300*17, 513)).transpose()
    testinstr1 = np.float64(np.reshape(test_instr1, (300*17, 513))).transpose()
    testinstr2 = np.float64(np.reshape(test_instr2, (300*17, 513))).transpose()
    intr1 = librosa.core.istft(testinstr1)
    intr2 = librosa.core.istft(testinstr2)
    mp31 = librosa.core.istft(out1)
    mp32 = librosa.core.istft(out2)
    librosa.output.write_wav('test_out1.wav', mp31, 22050)
    librosa.output.write_wav('test_out2.wav', mp32, 22050)
    librosa.output.write_wav('test1.wav', intr1, 22050)
    librosa.output.write_wav('test2.wav', intr2, 22050)
