import bson_fields_extractor
import tweepy
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

'''
# Function to collect data from Twitter API
def collect_data_from_twitter(api_key, api_secret_key):
    auth = tweepy.AppAuthHandler(api_key, api_secret_key)
    api = tweepy.API(auth)

    tweets = []
    for tweet in tweepy.Cursor(api.search, q='cyberattack', lang='en').items(1000):
        tweets.append(tweet.text)

    return tweets

# Function to collect data from external links
def collect_data_from_links(links):
    texts = []
    for link in links:
        response = requests.get(link)
        if response.status_code == 200:
            texts.append(response.text)

    return texts
'''

def load_tweets_bson(file_path):
    tweets_data = []
    with open(file_path, 'rb') as file:
        bson_data = file.read()
        while bson_data:
            record_length = bson_data[:4]  # First 4 bytes indicate the length of the BSON document
            document_length = int.from_bytes(record_length, 'little')  # Convert bytes to integer
            record = bson_fields_extractor.loads(bson_data[:document_length])  # Decode BSON document
            bson_data = bson_data[document_length:]  # Move to the next BSON document
            if 'text' in record:  # Assuming 'text' field contains the tweet text
                tweets_data.append(record['text'])
    return tweets_data

'''
# Collect data from Twitter API and external links
twitter_data = collect_data_from_twitter('your_api_key', 'your_api_secret_key')
link_data = collect_data_from_links(['link1', 'link2', 'link3'])
'''

# Combine the collected data
# data = twitter_data + link_data
# labels = ['malware'] * len(twitter_data) + ['phishing'] * len(twitter_data) + ['spam'] * len(twitter_data) + ['bot'] * len(twitter_data)

# Load data from files tweets.bson and include metadata from tweets.metadata.json
tweets_bson_data = load_tweets_bson('tweets.bson')  # Implement a function to load BSON data
metadata_json = '{"options":{},"indexes":[{"v":2,"key":{"_id":1},"name":"_id_","ns":"threat.tweets"},{"v":2,"unique":true,"key":{"id":1},"name":"id_1","ns":"threat.tweets"},{"v":2,"unique":true,"key":{"text":1},"name":"text_1","ns":"threat.tweets"}],"uuid":"cc49ecec23664a42a2559a2d61a1b4bd"}'

# Combine the collected data
# data = twitter_data + link_data + tweets_bson_data
data = tweets_bson_data
labels = []
categories = ['malware', 'phishing', 'spam', 'bot', 'unknown']
# Assign labels to each data sample
for category in categories:
    labels.extend([category] * len(tweets_bson_data))

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
X_seq = tokenizer.texts_to_sequences(data)
X_pad = pad_sequences(X_seq, maxlen=100)

# Convert labels to numerical values
label_dict = {category: i for i, category in enumerate(categories)}
y = np.array([label_dict[label] for label in labels])
print("X_pad shape:", X_pad.shape)
print("y shape:", y.shape)
'''
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
# Build the CNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=100))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(4, activation='softmax'))  # Output layer with 4 classes

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
'''