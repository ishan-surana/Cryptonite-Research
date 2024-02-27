import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load data from CSV file
data = pd.read_csv('tweets_final.csv')

# Extract relevant columns
tweets_data = data['text']
# data['type'] = data['type'].str.replace(r"[\[\]']", '', regex=True)
categories = data['type']

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data)
X_seq = tokenizer.texts_to_sequences(tweets_data)
X_pad = pad_sequences(X_seq, maxlen=100)

# Process text features
text_length = np.array([len(tweet) for tweet in tweets_data])
num_words = np.array([len(tweet.split()) for tweet in tweets_data])
num_chars = np.array([sum(c.isalpha() for c in tweet) for tweet in tweets_data])

# Process user features
user_features = pd.json_normalize(data['tweet'].apply(lambda x: eval(x)['user']))
user_screen_name = user_features['screen_name']
num_followers = user_features['followers_count']
num_friends = user_features['friends_count']
num_tweets = user_features['statuses_count']
user_creation_date = pd.to_datetime(user_features['created_at'], errors='coerce')  # Specify error handling
user_verified_status = user_features['verified'].astype(int)

# Convert user features to numpy arrays
user_features_array = user_features.to_numpy()
user_screen_name_array = user_screen_name.to_numpy()

# Process hashtag features
hashtags = data['tweet'].apply(lambda x: eval(x)['entities']['hashtags'])
num_hashtags = np.array([len(h) for h in hashtags])
hashtag_list = [' '.join([tag['text'] for tag in h]) for h in hashtags]

# Process URL features
urls = data['urls']
num_urls = np.array([len(u) for u in urls])
url_list = [' '.join(u) for u in urls]

# Process source features
source = data['tweet'].apply(lambda x: eval(x)['source'])

# Other features
timestamp = pd.to_datetime(data['date'], errors='coerce')  # Specify error handling
language = data['tweet'].apply(lambda x: eval(x)['lang'])
location = user_features['location']  # Assuming location is part of user details

# Add 'destination_url' and 'valid_certificate' features
destination_urls = data['destination_url'].fillna('NA')
valid_certificates = data['valid_certificate'].fillna('NA')

# Combine all features
X_combined = np.column_stack((X_pad, text_length, num_words, num_chars, num_followers, num_friends, num_tweets, user_features_array, user_screen_name_array, user_verified_status, num_hashtags, num_urls))

# Append additional features to X_combined
additional_features = np.column_stack((hashtags, url_list, source, timestamp, language, location))
X_combined = np.column_stack((X_combined, additional_features))

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