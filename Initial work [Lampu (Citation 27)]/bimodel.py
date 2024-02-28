import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate

# Load the SMS spam dataset
sms_df = pd.read_csv('datasets/sms_spam.csv')
# Load the email spam dataset
email_df = pd.read_csv('datasets/email_spam.csv')

# Upsample the email spam dataset to match the size of the SMS spam dataset
email_df_upsampled = resample(email_df, replace=True, n_samples=len(sms_df), random_state=42)

# Combine both datasets
combined_df = pd.concat([sms_df, email_df_upsampled], ignore_index=True)

# Tokenize the text data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(combined_df['text'])
text_sequences = tokenizer.texts_to_sequences(combined_df['text'])
max_len = 100
text_sequences_padded = pad_sequences(text_sequences, maxlen=max_len)

# Convert labels to binary (0s and 1s)
combined_df['type'] = (combined_df['type'] == 'spam').astype(int)
labels = combined_df['type'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    text_sequences_padded, labels, test_size=0.2, random_state=42
)

# Define the CNN model architecture
input_text = Input(shape=(max_len,))
embedding = Embedding(input_dim=max_words, output_dim=100, input_length=max_len)(input_text)
conv1d = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1)(embedding)
pooling = GlobalMaxPooling1D()(conv1d)
dense_text = Dense(128, activation='relu')(pooling)
dropout = Dropout(0.5)(dense_text)
output = Dense(1, activation='sigmoid')(dropout)  # Binary classification with sigmoid activation

# Define the model
model = Model(inputs=input_text, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
# Custom testing
print("\n\n\nCustom testing. Type 'exit' to exit.\n")
while(1):
    sample_message = str(input("Enter message:- "))
    if sample_message == "exit":
        break
    # Tokenize and pad the sample message
    sample_seq = tokenizer.texts_to_sequences([sample_message])
    sample_pad = pad_sequences(sample_seq, maxlen=max_len)
    # Predict the class of the sample message
    prediction = model.predict(sample_pad)
    # Interpret the prediction result
    print(f"\nPrediction = {prediction}")
    if prediction > 0.5:
        print("Sample message is classified as: spam\n")
    else:
        print("Sample message is classified as: ham\n")
'''