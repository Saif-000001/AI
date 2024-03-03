# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Function to extract mel spectrograms from audio files
# def extract_spectrogram(file_path, n_mels=128, hop_length=512):
#     y, sr = librosa.load(file_path, sr=None)  # Add sr=None to prevent resampling
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
#     mel_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_db

# # Load and preprocess the dataset
# data_path = "dataset_folder"
# labels = os.listdir(data_path)

# X, y = [], []

# for label_id, label in enumerate(labels):
#     genre_path = os.path.join(data_path, label)
    
#     for audio_file in os.listdir(genre_path):
#         file_path = os.path.join(genre_path, audio_file)
#         spectrogram = extract_spectrogram(file_path)
#         X.append(spectrogram)
#         y.append(label_id)

# X = np.array(X)
# y = np.array(y)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build the CNN model
# model = models.Sequential()
# model.add(layers.Input(shape=X_train[0].shape))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(len(labels), activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc}")

# # Make predictions on the test set
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Calculate accuracy using sklearn
# accuracy = accuracy_score(y_test, y_pred_classes)
# print(f"Sklearn Test Accuracy: {accuracy}")

# # Save the model for future use
# model.save("music_genre_cnn_model.h5")


