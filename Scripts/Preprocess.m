% Start EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Define the base path to your subjects' directories
basepath = '/home/technoshr370/Downloads/ds003751';

% Find all subject directories
subject_dirs = dir(fullfile(basepath, 'sub-*')); % Adjust pattern if needed

% Loop through all subject directories
for i = 1:length(subject_dirs)
    subject_path = fullfile(basepath, subject_dirs(i).name, 'eeg');

    % Find .set files within the eeg folder of each subject
    set_files = dir(fullfile(subject_path, '*.set'));

    % Process each .set file found
    for j = 1:length(set_files)
        % Load the EEG dataset
        EEG = pop_loadset('filename', set_files(j).name, 'filepath', set_files(j).folder);
        eeglab redraw;

        % Filter the data between 1 and 40 Hz
        EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);

        % Downsample the data to 256 Hz for faster computation
        EEG = pop_resample(EEG, 256);

        % Automatically clean the data
        EEG = clean_rawdata(EEG, 5, [-1, 1], 0.8, 4, -1, -1);

        % Re-reference the data to average
        EEG = pop_reref(EEG, []);

        % Run ICA with GPU acceleration
        EEG = pop_runica(EEG, 'extended', 1, 'usegpu', 1);

        % Use ICLabel for automatic component classification
        EEG = pop_iclabel(EEG, 'default');

        % Automatically reject components classified as eye movements or muscle
        bad_comps = find(EEG.etc.ic_classification.ICLabel.classifications(:, 3) > 0.5 | ...
                        EEG.etc.ic_classification.ICLabel.classifications(:, 5) > 0.5);
        EEG = pop_subcomp(EEG, bad_comps, 0);

        % Save the preprocessed data as .edf in the same directory
        pop_writeeeg(EEG, 'filename', [set_files(j).name(1:end-4) '.edf'], 'filepath', set_files(j).folder, 'TYPE', 'EDF');

    end
end
