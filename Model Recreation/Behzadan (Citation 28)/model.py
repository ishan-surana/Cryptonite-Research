import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load data from CSV file
column_names = ['_id', 'date', 'id', 'relevant', 'text', 'tweet', 'type', 'watson', 'annotation']
data = pd.read_csv('tweets.csv', names=column_names)

# Extract relevant columns
tweets_data = data['text']
data['type'] = data['type'].str.replace(r"[\[\]']", '', regex=True)
categories = data['type']

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data)
X_seq = tokenizer.texts_to_sequences(tweets_data)
X_pad = pad_sequences(X_seq, maxlen=100)

# Convert labels to numerical values
label_dict = {'vulnerability': 0, 'ransomware': 1, 'ddos': 2, 'leak': 3, 'general': 4, '0day': 5, 'botnet': 6, 'all': 7}
y = np.array([label_dict[category] for category in categories])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=100))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(8, activation='softmax'))  # Output layer with 8 classes

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')