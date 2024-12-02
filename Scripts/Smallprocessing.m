% Start EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Define the path to the specific subject's directory
subject_path = '/home/technoshr370/Downloads/ds003751/sub-mit097/eeg';

% Find .set files within the eeg folder of the subject
set_files = dir(fullfile(subject_path, '*.set'));

% Process each .set file found
for j = 1:length(set_files)
    EEG = pop_loadset('filename', set_files(j).name, 'filepath', set_files(j).folder);
    eeglab redraw;

    % Filter the data between 1 and 40 Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);

    % Check if downsampling is needed
    if EEG.srate > 256
        EEG = pop_resample(EEG, 256);
    end

    % Automatically clean the data
    EEG = clean_rawdata(EEG, 5, [-1, 1], 0.8, 4, -1, -1);

    % Re-reference the data to average
    EEG = pop_reref(EEG, []);

    % Save the preprocessed data as .edf in the same directory
    pop_writeeeg(EEG, fullfile(set_files(j).folder, [set_files(j).name(1:end-4) '.edf']), 'TYPE', 'EDF');
end
