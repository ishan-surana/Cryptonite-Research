import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate

# Load the datasets
sms_df = pd.read_csv('datasets/sms_spam.csv')
email_df = pd.read_csv('datasets/email_spam.csv')

# Combine both datasets
combined_df = pd.concat([sms_df, email_df], ignore_index=True)

# Tokenize the text data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(combined_df['text'])
text_sequences = tokenizer.texts_to_sequences(combined_df['text'])
max_len = 100
text_sequences_padded = pad_sequences(text_sequences, maxlen=max_len)

# Convert labels to binary (0s and 1s)
combined_df['type'] = combined_df['type'].map({'spam': 1, 'ham': 0, 'not spam': 0})
labels = combined_df['type'].values

# Split the data into training and testing sets
x_train_text, x_test_text, y_train, y_test = train_test_split(
    text_sequences_padded, labels, test_size=0.2, random_state=42
)

# Define the CNN model architecture for text data
input_text = Input(shape=(max_len,))
embedding = Embedding(input_dim=max_words, output_dim=100, input_length=max_len)(input_text)
conv1d = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1)(embedding)
pooling = GlobalMaxPooling1D()(conv1d)
dense_text = Dense(128, activation='relu')(pooling)

# Add dense layers for classification
dense = Dense(64, activation='relu')(dense_text)
dropout = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(dropout)  # Binary classification with sigmoid activation

# Define the model
model = Model(inputs=input_text, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train_text, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test_text, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])