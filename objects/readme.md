EEG Signals have two characteristics: power and frequency. When we derive PSD(Power Spectral Density), it gives us power corresponding to each frequency value present in the EEG. So, frequency[i] will have a corresponding power as power[i]. 


real_eeg_freq and real_eeg_power are the power <-> freq PSDs of the sample edf file.
simulation_model_freq and simulation_model_power are the PSDs of simulations(Kinetic EEG Simulation) created from the edf file.