import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Input, Concatenate, Dropout, Flatten
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load data from CSV file
data = pd.read_csv('tweets_final.csv')

# Data Preprocessing

# Data Cleaning
def clean_text(text):
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', '', text)
    # Replace repetitive line breaks and blank spaces with only one
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove emoticons and emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text

data['text'] = data['text'].apply(clean_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

data['text'] = data['text'].apply(lemmatize_text)

# POS Tagging
def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

data['pos_tags'] = data['text'].apply(pos_tagging)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
X_seq = tokenizer.texts_to_sequences(data['text'])
X_pad = pad_sequences(X_seq, maxlen=100)

# Stopwords removal
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

data['text'] = data['text'].apply(remove_stopwords)

# Message Structure preprocessing
# Extract structural features (Table 1)
# Define the function to extract structural features (Table 1)
def extract_structural_features(text):
    # Implement feature extraction logic
    # For example, count the number of characters, words, punctuation marks, etc.
    num_characters = len(text)
    num_words = len(text.split())
    # Return features as a list
    return [num_characters, num_words]

# Apply the function to extract structural features and create a new column
data['structural_features'] = data['text'].apply(extract_structural_features)

# Define the function to replace specific text components with predefined tokens
def replace_text_components(text):
    # Implement text component replacement logic
    # For example, replace email addresses with 'email_nlp', replace mentioned users with 'at_user_nlp', etc.
    # Here's a simple example:
    text = text.replace('@', 'at_user_nlp')
    text = text.replace('#', '')  # Remove hashtags
    # Add more replacement rules as needed
    return text

data['text'] = data['text'].apply(replace_text_components)

# URL Structure preprocessing
# Extract URL characteristics (Table 2) from the destination_url column
def extract_url_features(url):
    # Extract domain suffix and registrant from the URL
    if pd.isna(url):
        return 'NA', 'NA'
    else:
        domain_suffix = url.split('.')[-1] if '/' in url else 'NA'
        registrant = url.split('/')[2].split('.')[-2] if '/' in url else 'NA'
        return [domain_suffix, registrant]  # Return as list

# Extract URL features from destination_url
data[['domain_suffix', 'registrant']] = data['destination_url'].apply(lambda url: pd.Series(extract_url_features(url)))

# Combine domain_suffix and registrant into a single string column for hashing
data['combined_features'] = data.apply(lambda x: [x['domain_suffix'], x['registrant']], axis=1)

# Replace specific URL components with predefined tokens
def replace_url_components(url):
    # Replace email addresses and mentioned users with predefined tokens
    replaced_url = re.sub(r'[\w\.-]+@[\w\.-]+', 'email_nlp', url)
    replaced_url = re.sub(r'@[\w\.-]+', 'at_user_nlp', replaced_url)
    return replaced_url

# Replace NaN values with an empty string in the 'destination_url' column
data['destination_url'].fillna('', inplace=True)
# Replace URL components with predefined tokens
data['resolved_urls'] = data['destination_url'].apply(replace_url_components)

# Vectorization

# Hashing Vector for content
hashing_vectorizer = HashingVectorizer(n_features=100)
X_hash = hashing_vectorizer.fit_transform(data['text'])

# Text Structure Vector
X_text_structure = np.array(data['structural_features'].tolist())

# Hash the combined features
hasher = FeatureHasher(n_features=1000, input_type='string')
X_url_structure_hashed = hasher.transform(data['combined_features'])

# Convert hashed features to array
X_url_structure_hashed = X_url_structure_hashed.toarray()

# Convert labels to numerical values
label_dict = {'vulnerability': 0, 'ransomware': 1, 'ddos': 2, 'leak': 3, 'general': 4, '0day': 5, 'botnet': 6, 'all': 7}
y = np.array([label_dict[category] for category in data['type']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
X_train_hash, X_test_hash, _, _ = train_test_split(X_hash, y, test_size=0.2, random_state=42)
X_train_text_structure, X_test_text_structure, _, _ = train_test_split(X_text_structure, y, test_size=0.2, random_state=42)
X_train_url_structure, X_test_url_structure, _, _ = train_test_split(X_url_structure_hashed, y, test_size=0.2, random_state=42)

# Build the CNN model
input_content = Input(shape=(100,), name='content_input')
input_text_structure = Input(shape=(2,), name='text_structure_input')
input_url_structure = Input(shape=(1000,), name='url_structure_input')

# Additional input layers for other features
embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=100)(input_content)
conv_layer = Conv1D(8, 8, activation='relu')(embedding)
dropout_layer = Dropout(rate=0.26)(conv_layer)
pooling_layer = MaxPooling1D(pool_size=2)(dropout_layer)
flattened_layer = Flatten()(pooling_layer)

# Concatenate all input layers
concatenated_inputs = Concatenate()([flattened_layer, input_text_structure, input_url_structure])

# Fully connected layers
dense1 = Dense(155, activation='relu')(concatenated_inputs)
dense2 = Dense(105, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(25, activation='relu')(dense3)
dense5 = Dense(25, activation='relu')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output = Dense(5, activation='softmax')(dense6)  # Adjusted output neurons to 5

# Define the model
model = Model(inputs=[input_content, input_text_structure, input_url_structure], outputs=output)

model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    x=[X_train, X_train_text_structure, X_train_url_structure],
    y=y_train,
    batch_size=32,
    epochs=5,
    validation_data=(
        [X_test, X_test_text_structure, X_test_url_structure],
        y_test
    )
)

# Evaluate the model
loss, accuracy = model.evaluate([X_test, X_test_text_structure, X_test_url_structure], y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

model.summary()