import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Emotions mapping based on RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# List of observed emotions to consider
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_path) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        features = []
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            features.append(np.mean(mfccs, axis=1))
        if chroma:
            chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
            features.append(np.mean(chroma, axis=1))
        if mel:
            mel_spec = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            features.append(np.mean(mel_spec, axis=1))
        return np.concatenate(features)

# Function to load data from the dataset
def load_data(data_dir, test_size=0.2):
    X, y = [], []
    for actor_folder in glob.glob(os.path.join(data_dir, "Actor_*")):
        for file in glob.glob(os.path.join(actor_folder, "*.wav")):
            file_name = os.path.basename(file)
            emotion_code = file_name.split("-")[2]
            emotion = emotions[emotion_code]
            if emotion in observed_emotions:
                features = extract_features(file)
                if len(features) > 0:
                    X.append(features)
                    y.append(emotion)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Define the directory where the data is stored
data_directory = "./speech-emotion-recognition-ravdess-data/"

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_data(data_directory, test_size=0.25)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Print detailed classification report
print(classification_report(y_test, y_pred))
# Print the shape of the training and testing datasets
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Print the number of extracted features
print("Number of features extracted:", X_train.shape[1])

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

