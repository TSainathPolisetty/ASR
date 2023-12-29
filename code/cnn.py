# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:41:04 2023

@author: Tulasi Sainath Polisetty
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import librosa
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Define the path to your phenome and audio data
phenome_dir = "/home/tulasi/project/files/dataw/TrainFilesPhn"
audio_dir = "/home/tulasi/project/files/dataw/TrainFilesWav"

# Load audio data and phenome labels
def load_audio(file_path):
    # Load audio file
    audio, _ = librosa.load(file_path, sr=44100, mono=True)

    # Apply short-time Fourier transform
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

    # Convert to dB
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

    return spectrogram

# Load phenome labels
def load_phenome(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    phenome_seq = []
    for line in lines:
        phenome = line.split(' ')[2].rstrip()
        arpabet_code = arpabet.get(phenome)
        if arpabet_code is not None:
            phenome_seq.append(arpabet_code)
    return phenome_seq

# Define the mapping from phenomes to Arpabet codes
arpabet = {
    'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 6, 'axr': 7, 'ay': 8, 'b': 9,
    'bcl': 10, 'ch': 11, 'd': 12, 'dcl': 13, 'dh': 14, 'dx': 15, 'eh': 16, 'el': 17, 'em': 18,
    'en': 19, 'eng': 20, 'epi': 21, 'er': 22, 'ey': 23, 'f': 24, 'g': 25, 'gcl': 26, 'h#': 27,
    'hh': 28, 'hv': 29, 'ih': 30, 'ix': 31, 'iy': 32, 'jh': 33, 'k': 34, 'kcl': 35, 'l': 36,
    'm': 37, 'n': 38, 'ng': 39, 'nx': 40, 'ow': 41, 'oy': 42, 'p': 43, 'pau': 44, 'pcl': 45,
    'q': 46, 'r': 47, 's': 48, 'sh': 49, 't': 50, 'tcl': 51, 'th': 52, 'uh': 53, 'uw': 54,
    'ux': 55, 'v': 56, 'w': 57, 'y': 58, 'z': 59, 'zh': 60
}

train_data = []
train_labels = []
for label_file in os.listdir(phenome_dir):
    file_path = os.path.join(phenome_dir, label_file)
    phenome_seq = load_phenome(file_path)
    if len(phenome_seq) > 0:
        audio_file_path = os.path.join(audio_dir, label_file.replace(".PHN", ".WAV.wav"))
        spectrogram = load_audio(audio_file_path)        
        train_data.append(spectrogram)       
        train_labels.append(phenome_seq)
   
# Pad the input spectrograms
# Find the length of the longest spectrogram
max_length = 0
for spec in train_data:    
    if spec.shape[1] > max_length:      
        max_length = spec.shape[1]

pad_value = 0

# Iterate over each array and pad only the second column
padded_spec = []
for arr in train_data:
    # print(arr.shape)
    # Get the length of the second column
    col_len = arr.shape[0]
    # Pad the second column only and stack with the first column
    padded_arr = np.column_stack((arr[:, 0], pad_sequences(arr[:, 1:], padding='post', dtype='float32', value=pad_value, maxlen=max_length)))
    # Append the padded array to the list
    padded_spec.append(padded_arr)

# Pad the target labels
train_labels = pad_sequences(train_labels, padding='post')  

train_data = np.array(padded_spec)
train_labels = np.array(train_labels)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

y_train = y_train[..., np.newaxis]
y_val = y_val[..., np.newaxis]
y_train = np.expand_dims(y_train, axis=-2)
y_val = np.expand_dims(y_val, axis=-2)


# Reshape the input data for the CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Define the CNN model
print("model start")


# model = tf.keras.Sequential([
#     layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(y_train.shape[1], activation='relu'),  # Change the number of units to 61
#     layers.Reshape((y_train.shape[1], 1)),
#     layers.TimeDistributed(layers.Dense(len(arpabet), activation='softmax'))
# ])

model = tf.keras.Sequential([
    layers.Conv2D(2, kernel_size=(3, 3), input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])),
    layers.MaxPooling2D(pool_size=(8, 8)),
    layers.Flatten(),
    layers.Dense(2, activation='relu'),
    layers.Dense(y_train.shape[1], activation='relu'),  
    layers.Reshape((y_train.shape[1], 1)),
    layers.TimeDistributed(layers.Dense(len(arpabet), activation='softmax'))
])


model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


print("######################################")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print(len(arpabet))

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=8)


model.save("speech_recognition_model.h5")



# Print final training & validation accuracy values
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print("Final Training Accuracy: {:.2f}%".format(final_train_accuracy * 100))
print("Final Validation Accuracy: {:.2f}%".format(final_val_accuracy * 100))

# Print final training & validation loss values
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]    
print("Final Training Loss: {:.4f}".format(final_train_loss))
print("Final Validation Loss: {:.4f}".format(final_val_loss))


# X_train shape: (1718, 1025, 672, 1)
# y_train shape: (626, 1025, 672, 1, 1)
# X_val shape: (1718, 75, 1)
# y_val shape: (626, 73, 1, 1)

 # Plot training & validation accuracy values
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
