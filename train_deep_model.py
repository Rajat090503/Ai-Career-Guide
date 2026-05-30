import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Dataset
data = pd.read_csv("role_dataset.csv")

X = data["resume_text"]
y = data["role"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)

X_vectorized = vectorizer.fit_transform(X).toarray()

# Encode Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# Deep Learning Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_vectorized.shape[1],)),
    Dense(64, activation="relu"),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_vectorized,
    y_categorical,
    epochs=50,
    batch_size=4,
    verbose=1
)

# Save Model
model.save("deep_role_model.h5")

# Save Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save Label Encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Deep Learning Model Saved Successfully")