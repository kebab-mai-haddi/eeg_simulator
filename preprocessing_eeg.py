from scipy.signal import butter, lfilter
from pprint import pprint as pr
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import numpy as np
import warnings
import glob
import mne
import os
plt.style.use('ggplot')
warnings.filterwarnings("ignore")


recordings = []
plt.rcParams["figure.figsize"] = [8, 8]
# foll cols are not required, ch_to_drop states such columns.
ch_to_drop = ['COUNTER', 'INTERPOLATED', 'RAW_CQ', 'GYROX', 'GYROY', 'MARKER', 'MARKER_HARDWARE', 'SYNC', 'TIME_STAMP_ms',
              'CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 'CQ_P7', 'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4',
              'CQ_F8', 'CQ_AF4', 'CQ_CMS', 'STI 014', 'TIME_STAMP_s']
#not used
filters = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha':  (
    8, 13), 'beta':  (13, 30), 'gamma':  (30, 50)}
error_files = []


def preproecess_eeg(location='emotiv_recordings/utkarsh.edf'):
    recording = location
    # reading edf file, the eeg recordings using mne lib
    raw = mne.io.read_raw_edf(recording, preload=True, verbose=0)
    # dropping unwanted cols
    raw.drop_channels(ch_to_drop)
    # position of electrodes (all since artifacts are to be removed from all) in mne lib is 1020, the correct electrode name
    montage = mne.channels.read_montage('standard_1020')
    # assigns the above loaded electrodes position
    raw.set_montage(montage)

    raw_tmp = raw.copy()
    # will consider frequencies above 1Hz
    raw_tmp.filter(1, None, fir_design="firwin")
    # n x n matrix -> weight matrix in which features are extracted. features in terms of ML
    ica = mne.preprocessing.ICA(
        method="extended-infomax", random_state=1, verbose=0)
    # now the raw_tmp is fit into ica, the feature matrix so that we can get artifacts which are to be removed.
    ica.fit(raw_tmp)
    # looking at the plots generated from below we see the red areas nearby nose and conclude them as the eye movements
    ica.plot_components()
    # this particular eeg is having 3(muscle) 4 (ekg)and 13 (eye)
    ica.exclude = [3, 4, 13]
    # now data is cleansed.
    raw_corrected = raw.copy()
    # we have corrected ica, the weight matrix and now we are gonna apply it on raw_corrected.
    ica.apply(raw_corrected)

    temp = raw_corrected.copy()
    temp.pick_channels(['O1', 'O2'])
    return temp


# visualise the fie: we have 2 arrays in edf files, one arr for O1 and O2 each.
# get cclean eeg potentials for O1 and O2 electrodes.
datas = preproecess_eeg()
# get data in numpy array, list of lists : 1 fr O1 and 2 for O2
datas = datas.get_data()


for i, data in enumerate(datas):
    # eeg recordings were set at 128 Hz and so the sampling freq should be 128 in welch function.
    f, Pxx_spec = signal.welch(data, 128)
    np.save('objects/real_eeg_power_'+str(i)+'.npy', Pxx_spec)
    np.save('objects/real_eeg_freq.npy'+str(i)+'.npy', f)
    plt.figure()
    plt.semilogy(f, Pxx_spec, linewidth=2.0)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectrum (scipy.signal.welch)')
    plt.show()
