EEGLAB is a MATLAB-based toolbox commonly used for processing EEG data, including `.fdt` files. Here’s a step-by-step guide to work with `.fdt` files in EEGLAB:

---

### **What is an `.fdt` file?**

- `.fdt` (Fast Data Transport) is a binary file format used in EEGLAB to store EEG data.
- It is typically paired with a `.set` file, which contains metadata (channel names, sampling rate, etc.).

---

### **Steps to Process `.fdt` Files in EEGLAB**

#### **1. Install and Set Up EEGLAB**

1. Download EEGLAB from the [EEGLAB website](https://sccn.ucsd.edu/eeglab/index.php).
2. Launch MATLAB and add EEGLAB to the MATLAB path.
3. Start EEGLAB by typing `eeglab` in the MATLAB command window.

---

#### **2. Load the EEG Dataset**

- `.fdt` files are not standalone and require the corresponding `.set` file.
- To load the dataset:
    1. Go to the **File** menu in EEGLAB.
    2. Select **Load existing dataset**.
    3. Choose the `.set` file corresponding to your `.fdt` file.

---

#### **3. Explore the Data**

- Once loaded, EEGLAB will display the dataset in the main interface.
- To view the EEG data:
    - Click **Plot** > **Channel data (scroll)**.
    - You can inspect raw EEG signals here.

---

#### **4. Preprocess the Data**

Common preprocessing steps include:

##### **a. Filtering**

- Use **Tools** > **Filter the data** > **Basic FIR filter**.
- Apply a bandpass filter (e.g., 1–50 Hz) to remove noise and retain relevant EEG frequencies.

##### **b. Channel Locations**

- If the dataset lacks channel location information, go to **Edit** > **Channel locations** and load a standard montage (e.g., `standard-10-20-cap385.elp`).

##### **c. Artifact Rejection**

- **Manual rejection**: Use **Tools** > **Reject data epochs** > **Reject by inspection**.
- **Automatic rejection**: Use **Tools** > **Reject data epochs** > **Threshold rejection**.

##### **d. ICA for Artifact Removal**

- Perform ICA by going to **Tools** > **Run ICA**.
- After decomposition, identify and reject components associated with artifacts (e.g., eye blinks, muscle noise).

##### **e. Epoching**

- If your study involves event-related analysis, use **Tools** > **Extract epochs**.
- Select events and define time windows around them (e.g., -200 ms to 800 ms).

---

#### **5. Save the Processed Data**

- Save the dataset at any point:
    - **File** > **Save current dataset as**.
    - Save as a new `.set` file, which will automatically generate a corresponding `.fdt` file.

---

### **Example: Loading and Plotting Data**

Here’s an example script in MATLAB to load and visualize a dataset:

```matlab
% Start EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Load the .set file
EEG = pop_loadset('filename', 'example.set', 'filepath', 'path_to_file/');

% Check the dataset
eeg_checkset(EEG);

% Plot the raw data
pop_eegplot(EEG, 1, 1, 1);
```

---

### **Tips for `.fdt` Files**

1. **Missing `.set` File**: If you only have an `.fdt` file:
    
    - Recreate the `.set` file manually, specifying details such as sampling rate, channel names, and data dimensions. This is complex and not recommended unless necessary.
2. **Corrupt `.fdt` File**:
    
    - Check file integrity.
    - Use MATLAB’s `fopen` and `fread` functions to manually inspect the binary data if troubleshooting is required.



# EDF
Yes, you can convert preprocessed `.fdt` files to the widely-used **European Data Format (EDF)** using EEGLAB or custom MATLAB scripts. Here's how to achieve the conversion:

---

### **1. Using EEGLAB (Recommended)**

#### **Step 1: Load the `.fdt` File**

1. Open EEGLAB in MATLAB.
2. Load the dataset (`.set` file) associated with the `.fdt` file:
    - **File** > **Load existing dataset** > Select the `.set` file.

#### **Step 2: Export to EDF**

1. After preprocessing the data:
    - Go to **File** > **Export** > **Data file**.
    - Choose **EDF (European Data Format)**.
    - Provide the file name and save location.

---

### **2. Using a MATLAB Script**

If you want more control or automate the process, you can use MATLAB and a library like **Biosig** to convert the `.fdt` data to EDF format.

#### **Step 1: Install Biosig**

1. Download the **Biosig toolbox** from [Biosig's GitHub page](https://github.com/biosig).
2. Add the Biosig directory to the MATLAB path.

#### **Step 2: Load and Export Data**

Here’s an example MATLAB script to load `.fdt` files and save them as EDF:

```matlab
% Load EEGLAB and the dataset
[ALLEEG, EEG, CURRENTSET] = eeglab;
EEG = pop_loadset('filename', 'example.set', 'filepath', 'path_to_file/');

% Ensure dataset is clean and check it
EEG = eeg_checkset(EEG);

% Convert to EDF using Biosig
edf_filename = 'output_file.edf';
eeg_data = EEG.data; % EEG data matrix (channels x time points)

% Define EDF header information
hdr = struct();
hdr.SampleRate = EEG.srate; % Sampling rate
hdr.Label = {EEG.chanlocs.labels}; % Channel labels
hdr.PhysDim = 'uV'; % Physical unit
hdr.Type = 'EEG'; % Data type

% Save as EDF
savenewedf(edf_filename, eeg_data', hdr); % Transpose data for EDF format
disp(['Saved EEG data to ', edf_filename]);
```

---

### **3. Notes and Tips**

- **Channel Metadata**: Ensure channel names, sampling rates, and units are well-defined before exporting.
- **EDF Limitations**: EDF format supports fixed metadata like a maximum of 16 characters for channel labels. Adjust long labels accordingly.
- **Validation**: Use tools like EDFbrowser (free software) to inspect the resulting EDF file and confirm its correctness.

---

Would you like assistance with specific steps, such as setting up Biosig or troubleshooting the process?