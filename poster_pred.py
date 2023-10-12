import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Constants
POSTER_DIR = '/Users/geetika/Desktop/Poster Submission/poster_downloads'  # Modify this
CSV_PATH = '/Users/geetika/Desktop/Poster Submission/poster_scores.csv'
IMG_SIZE = (150, 150)  # Reduce this if you run into memory issues

# Load data
data = pd.read_csv(CSV_PATH)
posters = []
scores = []

for index, row in data.iterrows():
    img_path = os.path.join(POSTER_DIR, f"{row['IMDB_Score']}_{row['IMDB_ID']}.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    posters.append(img)
    scores.append(row['Poster_Score'])

posters = np.array(posters, dtype=np.float32) / 255.0  # Normalize images
scores = np.array(scores, dtype=np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(posters, scores, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)  # Regression, hence no activation in the last layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Saving model
model.save('poster_score_predictor.h5')

# To predict a score for a new poster in the future:
# new_img = ...  # Load and preprocess the new poster
# predicted_score = model.predict(np.expand_dims(new_img, axis=0))
