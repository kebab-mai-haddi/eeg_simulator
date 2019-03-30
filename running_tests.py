import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

m_f = np.load('objects/simulation_model_freq.npy')[:50]
m_p = np.load('objects/simulation_model_power.npy')[:50]
eeg_f = np.load('objects/real_eeg_freq.npy0.npy')[:50]
eeg_p = np.load('objects/real_eeg_power_0.npy')[:50]


plt.figure()
plt.semilogy(eeg_f, eeg_p,linewidth=2.0,c = 'b')
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()
