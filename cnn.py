from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Setting the random seed
np.random.seed(42)
tf.random.set_seed(42)

path = "C:/Users/BHARAT/Desktop/data sets/other_data/compressed_data/devnagri images/data.csv"
data = pd.read_csv(path)


label_encoder = LabelEncoder()
data['character'] = label_encoder.fit_transform(data['character'])
X = data.drop('character', axis=1).values.reshape(-1, 32, 32, 1).astype('float')/255
y = data['character']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2), strides=(3, 1)),
    BatchNormalization(),
#    Dropout(0.1),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=(1, 3)),
    BatchNormalization(),
#    Dropout(0.1),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=(3, 1)),
    BatchNormalization(),
#    Dropout(0.1),
    Flatten(),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dropout(0.1),
    BatchNormalization(),
    Dense(46, activation='softmax')
    ])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=early)
dump(model, 'devnagri.joblib')
dump(label_encoder, 'labels.joblib')
load = load('devnagri.joblib')


test_loss, test_accuracy = load.evaluate(X_test, y_test)
print(test_accuracy)
pred = load.predict(X_test)
prediction = pred.argmax(axis=1)
print(accuracy_score(prediction, y_test))
# Reverse encoding for predicted labels
predicted_labels = label_encoder.inverse_transform(prediction)

# Reverse encoding for true labels
true_labels = label_encoder.inverse_transform(y_test)

# Flatten the arrays
predicted_labels_flat = predicted_labels.flatten()
true_labels_flat = true_labels.flatten()

# Print classification report
print(classification_report(predicted_labels_flat, true_labels_flat))

