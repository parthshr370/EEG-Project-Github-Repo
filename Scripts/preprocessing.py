import mne 
import cupy as cp 
import numpy as np 

from mne.preprocessing import ICA 

# Load the EEG data
file_path = '/home/technoshr370/Downloads/Dataset/ds003751/sub-mit080/eeg/sub-mit080_task-Emotion_eeg.set'
raw = mne.io.read_raw_eeglab(file_path, preload=True)


raw.set_eeg_reference('average', projection=True)  # set to 'average' for average referencing

raw.apply_proj()

raw.filter(1., 40., method='iir', iir_params={'order': 5, 'ftype': 'butter'})

events, event_id = mne.events_from_annotations(raw)
event_id = {'click': 1}  # update based on your specific annotations and needs
tmin, tmax = -6, 1  # 6s before and 1s after the event
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)



# Apply ICA for artifact removal
ica = ICA(n_components=15, random_state=97)
ica.fit(epochs)


ica.plot_components()  # View ICA components to identify artifacts visually

# Manually select components to exclude based on your inspection
ica.exclude = [0, 1]  # example of excluding components by indices after visual inspection

# Apply the ICA cleaning
epochs_clean = ica.apply(epochs)

# Save the cleaned epochs to a new file, if needed
epochs_clean.save('/home/technoshr370/Downloads/Scripts/mit80.fif', overwrite=True)

print("Processing complete. Cleaned data saved.")