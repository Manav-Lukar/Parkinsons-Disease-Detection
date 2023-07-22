# # Parkinson's disease detection using speech input
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.pipeline import make_pipeline
# from python_speech_features import mfcc
# import librosa
# import sounddevice as sd
# import wavio
#
# # Step 1: Load the dataset and preprocess
# # Load the labeled dataset for Parkinson's disease and speech characteristics
# labeled_data_file = 'parkinsons_telemonitoring.csv'
# labeled_data = pd.read_csv(labeled_data_file)
# dataset_features = labeled_data.iloc[:, 6:].values
# dataset_labels = labeled_data.iloc[:, -1].values
#
# # Convert continuous labels to binary (0 or 1)
# label_binarizer = LabelEncoder()
# dataset_labels = label_binarizer.fit_transform(dataset_labels)
#
# # Step 2: Record audio from microphone
# duration = 7
# sample_rate = 16000
#
# print("Recording audio...")
# audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
# sd.wait()  # Wait until recording is complete
# print("Audio recording complete.")
#
# # Save the recorded audio as a WAV file
# recorded_filename = 'Sound_Recording.wav'
# wavio.write(recorded_filename, audio, sample_rate, sampwidth=2)
#
# # Step 3: Extract features from the recorded audio
# signal, _ = librosa.load(recorded_filename, sr=sample_rate)
#
# # Extract features from the recorded audio (e.g., MFCC)
# extracted_features = mfcc(signal, sample_rate)
# dataset_features = dataset_features[:extracted_features.shape[0], :]
# features_with_additional = np.concatenate((extracted_features, dataset_features), axis=1)
# dataset_labels = dataset_labels[:extracted_features.shape[0]]
#
# # Split the labeled dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features_with_additional, dataset_labels, test_size=0.2, random_state=42)
#
# # Step 4: Model Training and Evaluation
# # Create a pipeline for preprocessing and classification
# pipeline = make_pipeline(
#     StandardScaler(),
#     RandomForestClassifier(n_estimators=100, random_state=42)
# )
#
# # Train the model
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# # Step 5: Predict Parkinson's disease presence using the trained model
# recorded_features = mfcc(signal, sample_rate)
# combined_features = np.concatenate((recorded_features, dataset_features), axis=1)
# prediction = pipeline.predict(combined_features)
#
# # Output the prediction
# if prediction[0] == 1:
#     print("Parkinson's disease is detected.")
# else:
#     print("No Parkinson's disease detected.")


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')

df = df.drop(['name'], axis=1)

le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])

scaler = StandardScaler()
X = scaler.fit_transform(df.drop(['status'], axis=1).values)
y = df['status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print('Accuracy: {:.2%}'.format(accuracy))
print('Confusion Matrix:\n', cm)

sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()