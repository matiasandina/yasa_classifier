import yaml
import os
import sys
import numpy as np
import pandas as pd
import glob
import datetime
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
# this is my custom version of sleep staging
from src.staging import SleepStaging
import pyarrow as pa
import pyarrow.parquet as pq
import json
from py_console import console

def read_yaml(filename):
  with open(filename, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
  return cfg


def get_files(target_folder):
  eeg_file = os.path.join(target_folder, "EEG.mat")
  emg_file = os.path.join(target_folder, "EMG.mat")
  label_file = os.path.join(target_folder, "labels.mat")
  
  file_list = ["EEG.mat", "EMG.mat", "labels.mat"]
  full_file_list = list(
    map(lambda x: os.path.join(target_folder, x), 
        file_list)
  )
  
  paths_are_files = list(
    map(lambda x: os.path.isfile(x),
        full_file_list)
  )
  if not all(paths_are_files):
    raise FileNotFoundError
  # load files
  eeg = loadmat(eeg_file)
  emg = loadmat(emg_file)
  label = loadmat(label_file)
  # yasa values for hypnogram
  #* -2  = Unscored
  #* -1  = Artefact / Movement
  #* 0   = Wake
  #* 1   = N1 sleep
  #* 2   = N2 sleep
  #* 3   = N3 sleep
  #* 4   = REM sleep
  # arrange into proper shape
  accusleep_dict = {
    1:4,#"R",
    2:0,#"W",
    3:2,#"N"
  }
  # reshape everything to singleton
  label = np.squeeze(label["labels"])
  eeg_array = np.squeeze(eeg["EEG"])
  emg_array = np.squeeze(emg["EMG"])
  # re-map the label values
  label_df = pd.DataFrame({"label": label})
  label_df["label"] = label_df["label"].map(accusleep_dict)
  label_array = label_df["label"].values
  return eeg_array, emg_array, label_array

def create_channel_map(eeg_array, config, end = "extend"):
  channel_map = {}
  n_channels = eeg_array.shape[0]
  n_named_chan = len(config['selected_channels'])
  if  n_named_chan < n_channels:
    print(f"Data has {n_channels} channels but only {n_named_chan} named channels")
    # TODO: add "prepend option"
    if end == "extend":
      print("Adding channel(s) for camera(s) at the end")
      cam_names = [f"cam{i}" for i in range(1, n_channels - n_named_chan + 1)]
      new_channel_names = list(config['channel_names'])
      new_channel_names.extend(cam_names)
      for key, value in enumerate(new_channel_names):
        channel_map[key] = value 
    return(channel_map)
  else:
    return(config['channel_names'])

def validate_file_lengths(directory_list, sf, epoch_sec):
    '''
    Validate that EEG, EMG, and label data lengths match.
    len(EEG) should equal len(EMG) and len(labels) should equal len(EEG_epochs) which should equal len(EMG_epochs).
    Updates or discrepancies are logged.
    '''
    file_dict = {}
    for directory in directory_list:
        try:
            eeg, emg, labels = get_files(directory)
            eeg_epochs = eeg.shape[0] / (sf * epoch_sec)
            emg_epochs = emg.shape[0] / (sf * epoch_sec)
            labels_len = len(labels)
            file_dict[directory] = {
                "folder": directory.parts[-3:],
                "EEG_shape": eeg.shape,
                "EEG_epochs": eeg_epochs,
                "EMG_shape": emg.shape,
                "EMG_epochs": emg_epochs,
                "labels_len": labels_len,
                "valid_lengths": eeg.shape[0] == emg.shape[0] and eeg_epochs == emg_epochs == labels_len
            }
        except Exception as e:
            file_dict[directory] = {"error": str(e)}
            continue
    return file_dict


def report_validation_results(file_dict):
    '''
    Processes the file_dict to report validation results.
    Prints detailed validation status for each data set.
    '''
    from rich.console import Console
    console = Console()
    for directory, data in file_dict.items():
        if "error" in data:
            console.print(f"[bold on red]Failed to process[/bold on red] [blue bold]{directory}[/blue bold]: {data['error']}")
        elif data["valid_lengths"]:
            console.print(f"✅ All data lengths match in [blue bold]{directory}[/blue bold]: Validation successful.")
        else:
            console.print(f"❌ Data length mismatch in [blue bold]{directory}[/blue bold]:")
            console.print(f"    EEG length: {data['EEG_shape'][0]}, EMG length: {data['EMG_shape'][0]}")
            console.print(f"    Calculated EEG epochs: {data['EEG_epochs']}, EMG epochs: {data['EMG_epochs']}, Labels length: {data['labels_len']}")


def process_data(directory, config):
    import mne
    from mne.io import RawArray
    scaler = RobustScaler(with_centering=True, 
                      with_scaling=True, 
                      unit_variance = False)
    try:
        eeg, emg, labels = get_files(directory)
        if config['scale_features']:
            eeg = scaler.fit_transform(eeg.reshape(-1, 1)).squeeze()
            emg = scaler.fit_transform(emg.reshape(-1, 1)).squeeze()

        print(f'Data read from {directory}')
        info = mne.create_info(["eeg", "emg"], config['sf'], ch_types='misc', verbose=False)
        raw_array = RawArray(np.vstack((eeg, emg)), info, verbose=False)
        
        sls = SleepStaging(raw_array, eeg_name="eeg", emg_name="emg")
        sls.fit(epoch_sec=config['epoch_sec'])
        print("✅ Fit default bands finished")
        features = sls.get_features()
        # add the labels to the features
        features['label'] = labels

        save_features(features, directory, config)
    except Exception as e:
        print(f"Failed to process {directory}: {str(e)}")

def save_features(features, directory, config):
    scaled = "scaled" if config['scale_features'] else 'raw'
    file_path = os.path.join(directory, f"{scaled}_features.parquet")
    # Convert features DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(features, preserve_index=False)
    # Serialize the config dictionary to JSON
    config_json = json.dumps(config)
    # Embedding metadata into the PyArrow Table
    metadata = {'config': config_json}
    table = table.replace_schema_metadata(metadata)
    # Write the table with metadata to a Parquet file
    pq.write_table(table, file_path)
    print(f"✅ Features saved with metadata to {file_path}")

def load_features(file_path, convert_to_df=False):
    # Read the Parquet file with PyArrow
    table = pq.read_table(file_path)
    # Convert PyArrow Table to DataFrame
    # Accessing metadata
    metadata = table.schema.metadata
    # Optionally, convert the JSON metadata back to a dictionary if needed
    config = json.loads(metadata[b'config'])
    print("Config loaded from metadata:", config)
    if convert_to_df:
      # it might be better to use polars anyway
      features = table.to_pandas()
      return features, config
    else:
      return table, config


def predict_mouse(mouse_folder, config, classifier_path):
    import mne
    from mne.io import RawArray
    import numpy as np
    import joblib

    eeg, emg, true_labels = get_files(mouse_folder)
    info = mne.create_info(["eeg", "emg"], config['sf'], ch_types='misc', verbose=False)
    raw_array = RawArray(np.vstack((eeg, emg)), info, verbose=False)
    sls = SleepStaging(raw_array, eeg_name="eeg", emg_name="emg")
    sls.fit(epoch_sec=config['epoch_sec'])
    # Load the classifier
    clf = joblib.load(classifier_path)
    # Ensure the feature set matches
    # Get features from the model and the current dataset
    model_features = set(clf.feature_name_)  # This should be the list of features the model expects
    current_features = set(sls.feature_name_)  # Current features calculated by SleepStaging
    # Find features that are in the current set but not in the model's set
    features_to_drop = current_features - model_features
    # Adjust the features in sls to match the classifier's features
    if features_to_drop:
        sls._features = sls._features.drop(columns=list(features_to_drop))
        sls.feature_name_ = sls._features.columns.tolist()
    # Perform prediction
    predicted_labels = sls.predict(path_to_model=classifier_path)
    return sls, true_labels

def plot_spectrogram(eeg, hypno, config):
  sf = config['sf']
  import yasa
  # upsample to data
  label_df = yasa.hypno_upsample_to_data(hypno,
  sf_hypno=1/config['epoch_sec'], 
  data=eeg, sf_data=sf)
  fig = yasa.plot_spectrogram(eeg,
                      hypno=label_df, 
                      win_sec = 10,
                      sf=sf,
                      # default is 'RdBu_r'
                      # cmap='Spectral_r',
                      # manage the scale contrast,     larger values better contrast
                      trimperc = 1)
  fig.show()

def plot_eeg_sample(eeg, n=1000):
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(eeg[1:n], "k-")
  plt.show()
  plt.close()

def evaluate(predicted_labels, true_labels, plot_cm=False):
  import yasa
  import matplotlib.pyplot as plt
  from sklearn.metrics import accuracy_score
  import seaborn as sns
  from sklearn.metrics import cohen_kappa_score
  from sklearn.metrics import confusion_matrix
  
  # convert labels to string
  predicted_labels_string = yasa.hypno_int_to_str(predicted_labels)
  true_labels_string = yasa.hypno_int_to_str(true_labels)
  # Get labels in the order that they will show in cm
  cm_labels = yasa.hypno_int_to_str(np.unique(true_labels))
  
  accuracy = accuracy_score(
    y_true = true_labels_string,
    y_pred = predicted_labels_string
    )
  
  cohen_kappa = cohen_kappa_score(
    predicted_labels, true_labels)
  print(f"Accuracy: {accuracy:.3f} -- Cohen's Kappa: {cohen_kappa:.3f}")
  cm = confusion_matrix(predicted_labels, true_labels)
  if plot_cm:
    ax=plt.subplot()
    sns.heatmap(cm/np.sum(cm), annot=True, 
                fmt='.2%', cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(cm_labels) 
    ax.yaxis.set_ticklabels(cm_labels)
    plt.show()
    plt.close()
  return accuracy, cm, cohen_kappa


def validate_file_exists(folder, pattern):
  files_match = glob.glob(os.path.join(folder, pattern))
  if not files_match:
    console.error(f"{pattern} not found in {folder}", severe=True)
    sys.exit()
  else:
    if len(files_match) > 1:
      console.warning(f"{pattern} in more than a signle file")
      print(files_match)
      console.error(f"Stopping for pattern fixing")
      sys.exit()
    # unlist using first element, assume only one match...
    filepath = files_match[0]
    return filepath

def normalize_ttl(ttl_matrix, method="max"):
  if method == "max":
    max_per_channel = ttl_matrix.max(axis=1, keepdims=True)
    # remove the zeros so we can devide
    max_per_channel = np.where(max_per_channel == 0, 1, max_per_channel)
    out = ttl_matrix / max_per_channel
    return(out)

def find_pulse_onset(ttl_file, ttl_idx, timestamps_file, buffer, round=False):
  """
  This function reads the ttl pulse file
  Subsets the ttl_file array on ttl_idx
  buffer is sf / 4
  Finds pulse onset by calling np diff and looking for the moments where np.diff is positive
  There's two ways to call this. You can either return the rounded down timestamp (round=True) 
  # or interpolate from the closest timestamp assuming constant sampling rate.
  Rerturns the timestamps according to sampling frequency (sf)
  """
  sf = 4 * buffer
  ttl_events = np.load(ttl_file)
  # todo find in config
  photo_events = ttl_events[ttl_idx, :].flatten()
  pulse_onset = np.where(np.diff(photo_events, prepend=0) > 0)[0]
  # get division and remainder
  div_array = np.array([divmod(i, buffer) for i in pulse_onset])
  # TODO div_array[:, 0] has the rounded version
  # timestamps.iloc[div_array[:, 0], :] + sampling_period * div_array[:, 1] is the way to calculate the proper timestamp
  # this assumes constant sampling rate between known timestamps
  timestamps = pd.read_csv(timestamps_file)
  # TODO: not sure this works for all sf 
  sampling_period_ms = 1/sf * 1000
  out = timestamps.iloc[div_array[:,0], :].copy()
  if round:
    return out
  else:
    out.iloc[:, 0] = out.iloc[:, 0] + sampling_period_ms * div_array[:, 1]
    dt = (sampling_period_ms * div_array[:, 1]).astype('timedelta64[ms]')
    out.iloc[:, 1] = pd.to_datetime(out.iloc[:,1]) + dt
    return out
