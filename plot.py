import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

loss_values = np.load('loss_values.npy')
with open('MIR_eval_values.pickle', 'rb') as handle:
    mir_eval = pickle.load(handle)

sir = mir_eval['SIR']
sdr = mir_eval['SDR']
sar = mir_eval['SAR']

plt.figure(1)
plt.plot(loss_values)
plt.ylabel('Epoch')
plt.xlabel('Loss value')
plt.show()

plt.figure(2)
plt_sar = plt.plot(sar, label='SAR')
plt_sdr = plt.plot(sdr, label='SDR')
plt.ylabel('Epoch')
plt.xlabel('db')
plt.legend(loc='upper left')
plt.show()
