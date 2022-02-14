# %% [markdown]
# ### Audio Classification Data Inspector, Feature Extraction Document
# In this document we will read in our data set:
# * read in all our audio data, train and test
# * inspect the diversity of the audio data
# * Then perform feature extraction and cache as a NPY file for MFCC and Mel-Spectrogram.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import helper
import os
import math 

# reproducibility
np.random.seed(42)

# %% [markdown]
# ### Dataset Read In
# In this section we will read in our data set structure which is located in our home holder
# 
# ```
# root:
#     kaggle_2018_dataset:
#         train:
#             wav files for training only, contains catalog.csv
#         test:
#             wav files for testing our model, contains catalog.csv
#         data:
#             npy files, feature extraction MFCC and MEL-SPEC
#         models:
#             trained models
# ```

# %%
# get the data set root directory path
dataset_root_dir = os.path.join("/home/charlesedwards/Documents", 'kaggle_2018_dataset')

# get the train and test data directories
train_dir = os.path.join(dataset_root_dir, 'train')
test_dir = os.path.join(dataset_root_dir, 'test')

# get the catalog.csv for train and test directories
train_catalog_csv = os.path.join(train_dir, 'catalog.csv')
test_catalog_csv = os.path.join(test_dir, 'catalog.csv')

# read the catalog.csv files
train_metadata = pd.read_csv(train_catalog_csv)
test_metadata = pd.read_csv(test_catalog_csv)

# drop unwanted columns 
train_metadata.drop(['license','freesound_id'], axis=1, inplace=True)
test_metadata.drop(['license','freesound_id'], axis=1, inplace=True)

# display the first 5 rows of both metadatas
helper.display_side_by_side([train_metadata.head(),test_metadata.head()], ['train_metadata', 'test_metadata'])

# %% [markdown]
# ### Visualize Label Distribution 
# In this section of the notebook we will display the label distribution i.e number of audio files per class

# %%
# visualize the label count distribution
plt.figure(figsize=(15,4))
chart = train_metadata['label'].value_counts().plot(kind='bar',)
chart.set_xticklabels(labels=chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title("Total number of audio files per class")
chart.set_ylabel('Count')

print('Minimum samples per category = ', min(train_metadata['label'].value_counts()))
print('Maximum samples per category = ', max(train_metadata['label'].value_counts()))

# %% [markdown]
# ### Visualize Verified Label Distribution 
# In this section of the notebook we will display how many labels are verified vs not

# %%
plt.figure(figsize=(4,3))

# plot the number of manually_verified vs non manually_verified audio clips
verified_count = train_metadata['manually_verified'].value_counts()
verified_count.plot(kind='bar', color=['green', 'red'], rot=0, title='Verified vs Non-Verified Audio Clips')


# %% [markdown]
# ### Read Audio To Memory
# This process will take a lot of ram and may vary on dataset size. Atm a 5GB dataset uses roughly 12GB of ram

# %%
# Windowing
n_fft=1024
hop_length=None#512

def load_data_set(data_dir, metadata_pd, sample_rate=None, max_duration=2.0, max_samples=None):

    processed_data = []
    sample_rates = []
    durations = []

    print(f"Processing audio from: {data_dir} ... with {len(metadata_pd)} files\nmax_duration: {max_duration}\nn_fft: {n_fft}\nhop_length: {hop_length}")

    length = len(metadata_pd)
    counter = 0

    # using librosa for every fname in our data_frame generate a mel spectrogram and save it to a numpy array
    for x, row in enumerate(metadata_pd.iloc):
        
        # load the audio file
        y, sr = librosa.load( os.path.join( data_dir, row['fname'] ), sr=sample_rate, duration=max_duration)

        # re sample the audio data to a lower sample rate
        #y_22k = librosa.resample(y, orig_sr=sr, target_sr=22050)

        # normalize the audio with librosa
        normalized_y = librosa.util.normalize(y)

        # append the data to our processed_data array
        processed_data.append(normalized_y)
        sample_rates.append(sr)
        durations.append(librosa.get_duration(y=y, sr=sr))

        # Notify update every N files
        if (counter == 500):
            print("Status: {}/{}".format(x+1, length))
            counter = 0

        counter += 1


    print("done")

    print("appending to processed_data as column 'data'")
    # append the processed_data to the data frame as data
    metadata_pd["data"] = processed_data
    print("done")

    print("appending to sample_rates as column 'sr'")
    # append the sample_rates to the data frame as sr
    metadata_pd["sr"] = sample_rates
    print("done")

    print("appending to durations as column 'duration'")
    # append the sample_rates to the data frame as sr
    metadata_pd["duration"] = durations
    print("done")


# %%
# load in the train data set
load_data_set(train_dir, train_metadata, max_duration=None)

# load in the test data set
load_data_set(test_dir, test_metadata, max_duration=None)

# %%
# display the first 5 rows of both metadatas again as we now have normalized audio data
helper.display_side_by_side([train_metadata.head(),test_metadata.head()], ['train_metadata', 'test_metadata'])

# %%
# get the average duration of all audio clips in the pd data frame
print("Average duration of all audio TRAIN clips: {}".format(np.mean(train_metadata['duration'])))

print("Average duration of all audio TEST clips: {}".format(np.mean(test_metadata['duration'])))

# get the lowest and highest duration of all audio clips in the pd data frame
print("Lowest duration of all audio TRAIN clips: {}".format(np.min(train_metadata['duration'])))

print("Highest duration of all audio TRAIN clips: {}".format(np.max(train_metadata['duration'])))

# %% [markdown]
# ### Data Visualization 1/3
# Visualize the train_metadata, in this section generate **STFT** spectrograms to db/log scale

# %%
# display 6 random spectrograms
# for i in range(6):

#     # compute a short-time Fourier transform (STFT)
#     D = librosa.stft(train_metadata.iloc[i]['data'], n_fft=n_fft, hop_length=hop_length)
#     #D = D.T # convert to frequency-time
#     # convert to db
#     D_db = librosa.amplitude_to_db(np.abs(D))

#     # display the spectrogram and data side by side
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(D_db, aspect='auto', origin='lower', cmap='gray_r')
#     plt.title(train_metadata.iloc[i]['label'])
#     plt.subplot(1, 2, 2)
#     plt.plot(D_db)
#     plt.show()


# %% [markdown]
# ### Data Visualization 2/3
# Display 6 audio **MFCCs** of the train_metadata,
# Mel-frequency cepstral coefficients

# %%
# https://arxiv.org/pdf/1908.05863.pdf



# # display 6 random MFCCS
# for i in range(6):

#     # num_segments = 5

#     # SAMPLE_RATE = train_metadata.iloc[i]['sr']
#     # TRACK_DURATION = train_metadata.iloc[i]['duration']#7 # measured in seconds
#     # SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

#     # num_mfcc=40#13
#     # n_fft=2048
#     # hop_length=512#64#128#256#512

#     # samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
#     # num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

#     # # process all segments of audio file
#     # for d in range(num_segments):

#     #     # calculate start and finish sample for current segment
#     #     start = samples_per_segment * d
#     #     finish = start + samples_per_segment

#     #     try:
#     #         # extract mfcc
#     #         #mfcc = librosa.feature.mfcc(train_metadata.iloc[i]['data'][start:finish], train_metadata.iloc[i]['sr'], n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
#     #         #mfcc = mfcc.T

#     #         D = librosa.stft(train_metadata.iloc[i]['data'][start:finish], n_fft=n_fft, hop_length=hop_length)
#     #         #D = D.T # convert to frequency-time
#     #         #    convert to db
#     #         mfcc = librosa.amplitude_to_db(np.abs(D))

#     #         print(mfcc.shape)

#     #         # display the spectrogram and data side by side
#     #         plt.figure(figsize=(10, 4))
#     #         plt.subplot(1, 2, 1)
#     #         plt.imshow(mfcc, aspect='auto', origin='lower', cmap="inferno")
#     #         plt.title(train_metadata.iloc[i]['label'])
#     #         plt.subplot(1, 2, 2)
#     #         plt.plot(mfcc)
#     #         plt.show()
#     #     except Exception as e:
#     #         print(f"Error processing segment {train_metadata.iloc[i]['fname']} {e}, probably not enough data to segment it")

#     #     # store only mfcc feature with expected number of vectors
#     #     #if len(mfcc) == num_mfcc_vectors_per_segment:
#     #         #data["mfcc"].append(mfcc.tolist())
#     #         #data["labels"].append(i-1)
#     #         #print("{}, segment:{}".format(file_path, d+1))


#     mfcc = librosa.feature.mfcc(train_metadata.iloc[i]['data'], sr=train_metadata.iloc[i]['sr'], n_mfcc=40)

#     # display the spectrogram and data side by side
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(mfcc, aspect='auto', origin='lower', cmap="inferno")
#     plt.title(train_metadata.iloc[i]['label'])
#     plt.subplot(1, 2, 2)
#     plt.plot(mfcc)
#     plt.show()

# %% [markdown]
# ### Data Visualization 3/3
# Display 6 audio **MEL-Spectrogram** of the train_metadata

# %%
# create MEL-scaled filter banks spectrograms
#n_mels = 128
# x_test_pad = 0 #* 2 # 2 seconds   

# # display 6 random mel spectrograms
# for i in range(6):

#     # generate a mel scaled spectrogram
#     mel_spectrogram = librosa.feature.melspectrogram(train_metadata.iloc[i]['data'], sr=train_metadata.iloc[i]['sr'], n_fft=1024) # n_mels=n_mels

#     # convert the sound intensity to log scale
#     mel_db = librosa.power_to_db(np.abs(mel_spectrogram)) #librosa.amplitude_to_db(np.abs(mel_spectrogram))

#     # normalize the data to 0-1
#     normalized_mel = librosa.util.normalize(mel_db)

#     # Should we require padding
#     shape = normalized_mel.shape[1]
#     if (x_test_pad > 0 & shape < x_test_pad):
#         xDiff = x_test_pad - shape
#         xLeft = xDiff//2
#         xRight = xDiff-xLeft
#         print("[WARNING] PADDING MEL-SPEC START")
#         normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')
#         print("[WARNING] PADDING MEL-SPEC END")

#     # plot the spectrogram and data side by side
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(normalized_mel, aspect='auto', origin='lower', cmap='plasma')
#     plt.title(train_metadata.iloc[i]['label'])
#     plt.subplot(1, 2, 2)
#     plt.plot(normalized_mel)
#     plt.show()


# %% [markdown]
# ### Begin Feature Extraction For MFCC for train and test datasets
# We will cache the feature extraction into npy files of the train and test datasets for training

# %%
def extract_mfcc_features(metadata_pd, n_mfcc=40, base_name=None):

    if base_name is None:
        print("base_name is None")
        return

    base_name = 'mfcc-' + base_name

    print(f"Starting Extraction of MFCC features for base name: {base_name} total, {len(metadata_pd)} files")

    # Iterate through all audio files and extract MFCC
    features = []
    labels = []
    frames_max = 0
    counter = 0
    total_samples = len(metadata_pd)
    mfcc_max_padding = 0

    # # segmentation code
    # num_segments = 5

    # SAMPLE_RATE = 44100
    # TRACK_DURATION = 7 # measured in seconds
    # SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    # num_mfcc=40#13
    # n_fft=2048
    # hop_length=512#64#128#256#512

    # samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for index, row in metadata_pd.iterrows():
        class_label = row["label"]

        num_segments = 5

        SAMPLE_RATE = row['sr']
        TRACK_DURATION = row['duration'] #7 # measured in seconds
        SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

        num_mfcc=n_mfcc#13
        n_fft=1024#2048
        hop_length=512#64#128#256#512

        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        #num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        # Extract MFCC data
        #mfcc = librosa.feature.mfcc(row['data'], sr=row['sr'], n_mfcc=n_mfcc)
        
        # normalize the mfcc between -1 and 1
        #normalized_mfcc = librosa.util.normalize(mfcc)

        # process all segments of audio file
        for d in range(num_segments):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            try:
                # extract mfcc
                mfcc = librosa.feature.mfcc(row['data'][start:finish], row['sr'], n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = librosa.util.normalize(mfcc) # normalize the data
                mfcc = mfcc.T # transpose the data to x:time y:frequency
  
                # Save current frame count
                num_frames = mfcc.shape[1]
                
                # Add row (feature / label)
                features.append(mfcc)
                labels.append(class_label)

                # Update frames maximum
                if (num_frames > frames_max):
                    frames_max = num_frames

                # # display the spectrogram and data side by side
                # plt.figure(figsize=(10, 4))
                # plt.subplot(1, 2, 1)
                # plt.imshow(mfcc, aspect='auto', origin='lower', cmap="inferno")
                # plt.title(train_metadata.iloc[i]['label'])
                # plt.subplot(1, 2, 2)
                # plt.plot(mfcc)
                # plt.show()
            except Exception as e:
                print(f"Error processing segment {row['fname']} {e}, probably not enough data to segment it")

        # # Should we require padding
        # shape = normalized_mfcc.shape[1]
        # if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
        #     xDiff = mfcc_max_padding - shape
        #     xLeft = xDiff//2
        #     xRight = xDiff-xLeft
        #     normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        #     print("[WARNING] PADDING MFCC")

    

        # Notify update every N files
        if (counter == 500):
            print("Status: {}/{}".format(index+1, total_samples))
            counter = 0

        counter += 1
        
    print("Finished: {}/{} frames_max {}".format(index, total_samples, frames_max))
    # Add padding to features with less than frames than frames_max
    padded_features = helper.add_padding(features, frames_max)
    # Verify shapes
    print("Raw features length: {}".format(len(features)))
    print("Padded features length: {}".format(len(padded_features)))
    print("Feature labels length: {}".format(len(labels)))

    # Convert features (X) and labels (y) to Numpy arrays
    X = np.array(padded_features)
    y = np.array(labels)

    data_npy_folder = os.path.join(dataset_root_dir, 'data')

    # Optionally save the features to disk
    np.save( os.path.join(data_npy_folder, f"X-{base_name}" ), X)
    np.save( os.path.join(data_npy_folder, f"y-{base_name}" ), y)

    # free up memory
    del X
    del y
    del features
    del labels
    
    print(f"Finished Extraction of MFCC features for base name: {base_name} total, {len(metadata_pd)} files")

# %%
extract_mfcc_features(train_metadata, n_mfcc=40, base_name='train')
extract_mfcc_features(test_metadata, n_mfcc=40, base_name='test')

# %% [markdown]
# ### Begin Feature Extraction For MEL for train and test datasets
# We will cache the feature extraction into npy files of the train and test datasets for training

# %%
def extract_mel_spectrogram_features(metadata_pd, n_fft=1024, base_name=None):

    if base_name is None:
        print("base_name is None")
        return

    base_name = 'mel-' + base_name

    print(f"Starting Extraction of MEL-Spec features for base name: {base_name} total, {len(metadata_pd)} files")

    # Iterate through all audio files and extract mel spectrograms
    features = []
    labels = []
    frames_max = 0
    counter = 0
    total_samples = len(metadata_pd)
    #mel_max_padding = 0

    for index, row in metadata_pd.iterrows():
        num_segments = 5

        SAMPLE_RATE = row['sr']
        TRACK_DURATION = row['duration'] #7 # measured in seconds
        SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

        #num_mfcc=n_mfcc#13
        n_fft=1024#2048
        hop_length=512#64#128#256#512

        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        #num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        # generate a mel scaled spectrogram
        #mel_spectrogram = librosa.feature.melspectrogram(row['data'], sr=row['sr'], n_fft=n_fft)#, n_mels=n_mels)

        # convert the sound intensity to log scale
        #mel_db = librosa.power_to_db(np.abs(mel_spectrogram)) #librosa.amplitude_to_db(np.abs(mel_spectrogram))

        # normalize the data to 0-1
        #normalized_mel = librosa.util.normalize(mel_db)

        # # Should we require padding
        # shape = normalized_mel.shape[1]
        # if (mel_max_padding > 0 & shape < mel_max_padding):
        #     xDiff = mel_max_padding - shape
        #     xLeft = xDiff//2
        #     xRight = xDiff-xLeft
        #     normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        #     print("[WARNING] PADDING MEL-SPEC")


        # process all segments of audio file
        for d in range(num_segments):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            try:

                # generate a mel scaled spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(row['data'][start:finish], sr=row['sr'], n_fft=n_fft, hop_length=hop_length)
                mel_spectrogram = librosa.power_to_db(np.abs(mel_spectrogram)) #librosa.amplitude_to_db(np.abs(mel_spectrogram))
                mel_spectrogram = librosa.util.normalize(mel_spectrogram)
                mel_spectrogram = mel_spectrogram.T # x:time y:frequency

                # Save current frame count
                num_frames = mel_spectrogram.shape[1]
                
                # Add row (feature / label)
                features.append(mel_spectrogram)
                labels.append(row["label"])

                # Update frames maximum
                if (num_frames > frames_max):
                    frames_max = num_frames

            except Exception as e:
                print(f"Error processing segment {row['fname']} {e}, probably not enough data to segment it")

        # Notify update every N files
        if (counter == 500):
            print("Status: {}/{}".format(index+1, total_samples))
            counter = 0

        counter += 1
        
    print("Finished: {}/{} frames_max {}".format(index, total_samples, frames_max))
    # Add padding to features with less than frames than frames_max
    padded_features = helper.add_padding(features, frames_max)
    # Verify shapes
    print("Raw features length: {}".format(len(features)))
    print("Padded features length: {}".format(len(padded_features)))
    print("Feature labels length: {}".format(len(labels)))

    # Convert features (X) and labels (y) to Numpy arrays
    X = np.array(padded_features)
    y = np.array(labels)

    data_npy_folder = os.path.join(dataset_root_dir, 'data')

    # Optionally save the features to disk
    np.save( os.path.join(data_npy_folder, f"X-{base_name}" ), X)
    np.save( os.path.join(data_npy_folder, f"y-{base_name}" ), y)

    # free up memory
    del X
    del y
    del features
    del labels
    
    print(f"Finished Extraction of MEL-Spec features for base name: {base_name} total, {len(metadata_pd)} files")

# %%
extract_mel_spectrogram_features(train_metadata, base_name='train')
extract_mel_spectrogram_features(test_metadata, base_name='test')


