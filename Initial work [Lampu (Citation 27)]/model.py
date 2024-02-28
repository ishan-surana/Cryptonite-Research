import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, TextVectorization

# Load the CSV file
df = pd.read_csv('datasets/sms_spam.csv')
'''
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
'''
# Split the data into training and testing sets
x = df['text'].values
y = df['type'].values
# Convert labels to binary (0s and 1s)
y = (y == 'spam').astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Tokenize the text data
max_words = 1000  # or any other value
tokenizer = Tokenizer(num_words=max_words)
'''
# Suggested by VS Code (as Tokenizer depreciated)
textvector = TextVectorization(max_tokens=max_words)
print(textvector)
'''
tokenizer.fit_on_texts(x_train)
X_train_seq = tokenizer.texts_to_sequences(x_train)
X_test_seq = tokenizer.texts_to_sequences(x_test)

# Pad the sequences
max_len = 100  # or any other value
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Define the CNN model architecture
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification, so 1 output node with sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, batch_size=32, epochs=10, validation_data=(X_test_pad, y_test))

# Evaluate the model
score = model.evaluate(X_test_pad, y_test, batch_size=32)
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