import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import ast
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import seaborn as sns

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_structural_features(text):
    message_length = len(text)
    num_tokens = len(word_tokenize(text))
    num_hashtags = text.count('#')
    num_emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    num_urls = text.count('http://') + text.count('https://')
    num_periods = text.count('.')
    num_commas = text.count(',')
    num_digits = sum(c.isdigit() for c in text)
    num_sentences = len(sent_tokenize(text))
    num_mentioned_users = text.count('@')
    num_uppercase = sum(c.isupper() for c in text)
    num_question_marks = text.count('?')
    num_exclamation_marks = text.count('!')
    emojis = set(re.findall(r'\:[\w]+\:', emoji.demojize(text)))
    num_emoticons = len(emojis)
    num_dollar_symbols = text.count('$')
    num_other_symbols = len([char for char in text if char not in '"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@.://,?!' + ''.join(emojis)])
    return [message_length, num_tokens, num_hashtags, num_emails, num_urls, num_periods, num_commas, num_digits, num_sentences, num_mentioned_users, num_uppercase, num_question_marks, num_exclamation_marks, num_emoticons, num_dollar_symbols, num_other_symbols]

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def replace_text_components(text):
    text = text.replace('.', '_period_')
    text = text.replace('/', '_slash_')
    text = text.replace('@', '_at_')
    text = text.replace('-', '_hyphen_')
    text = text.replace(':', '_colon_')
    text = text.replace('#', '')
    return text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def extract_first_url(url_list):
    try:
        urls = ast.literal_eval(url_list)
        first_url = urls[0] if urls else None
        return first_url
    except (SyntaxError, ValueError):
        return None

def extract_url_features(url, urls, certificate):
    if pd.isna(url):
        return ['NA'] * 24
    else:
        url_length = len(url)
        has_security_protocol = 1 if url.startswith(('http://', 'https://')) else 0
        first_url = extract_first_url(urls)
        is_shortened_url = 1 if first_url and len(url) < len(first_url) else 0
        strings_divided_by_periods = len(url.split('.'))
        strings_divided_by_hyphens = len(url.split('-'))
        strings_divided_by_slashes = len(url.split('/'))
        num_words = len(re.findall(r'\b\w+\b', url))
        num_ips = len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))
        num_digits = sum(c.isdigit() for c in url)
        num_hyphens = url.count('-')
        num_periods = url.count('.')
        num_slashes = url.count('/')
        num_uppercase = sum(c.isupper() for c in url)
        num_lowercase = sum(c.islower() for c in url)
        num_ampersand_symbols = url.count('&')
        num_equal_symbols = url.count('=')
        num_question_marks = url.count('?')
        num_wave_symbols = url.count('~')
        num_plus_signs = url.count('+')
        num_colon_symbols = url.count(':')
        num_other_characters = len([char for char in url if char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@.://,?!-'])
        has_extension = 1 if '.' in url else 0
        domain_suffix = url.split('/')[2].split('?')[0].split('#')[0].split('.')[-1] if '/' in url else 'NA'
        registrant = url.split('/')[2].split('?')[0].split('#')[0].split('.')[-2] if '/' in url else 'NA'
        if 'utm' in registrant:
            print(url)
        valid_certificate = 1 if certificate else 0
        return [url_length, has_security_protocol, is_shortened_url, strings_divided_by_periods, strings_divided_by_hyphens, strings_divided_by_slashes, num_words, num_ips, num_digits, num_hyphens, num_periods, num_slashes, num_uppercase, num_lowercase, num_ampersand_symbols, num_equal_symbols, num_question_marks, num_wave_symbols, num_plus_signs, num_colon_symbols, num_other_characters, has_extension, domain_suffix, registrant, valid_certificate]

def replace_url_components(url):
    replaced_url = re.sub(r'[\w\.-]+@[\w\.-]+', 'email_nlp', url)
    replaced_url = re.sub(r'@[\w\.-]+', 'at_user_nlp', replaced_url)
    return replaced_url

with open('Improvements/outputs/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('Improvements/outputs/url_hasher.pickle', 'rb') as handle:
    url_hasher = pickle.load(handle)
with open('Improvements/outputs/label_dict.pickle', 'rb') as handle:
    label_dict = pickle.load(handle)

model = load_model('Improvements/outputs/advanced_cnn_model.keras')

def summary_to_markdown(modelsummary):
    # Clean the model summary
    modelsummary = re.sub(r'=+', '', modelsummary)

    # Extract total, trainable, and non-trainable params
    total_params = re.search(r'Total params:\s+([\d,]+)', modelsummary).group(1)
    trainable_params = re.search(r'Trainable params:\s+([\d,]+)', modelsummary).group(1)
    non_trainable_params = re.search(r'Non-trainable params:\s+([\d,]+)', modelsummary).group(1)

    lines = modelsummary.strip().splitlines()
    markdown_lines = []
    markdown_lines.append("| Layer (type)                | Output Shape                 | Param #   | Connected to                                    |")
    markdown_lines.append("|-----------------------------|------------------------------|-----------|------------------------------------------------|")

    layer_buffer = ""
    output_shape_buffer = ""
    param_buffer = ""
    connected_to_buffer = ""

    for line in lines[3:-4]:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        parts = re.split(r'(?<!\s)\s{2,}', stripped_line)
        if len(parts) >= 4:
            if layer_buffer or output_shape_buffer or param_buffer or connected_to_buffer:
                markdown_lines.append(f"| {layer_buffer:<28} | {output_shape_buffer:<28} | {param_buffer:<9} | {connected_to_buffer:<48} |")
                layer_buffer = ""
                output_shape_buffer = ""
                param_buffer = ""
                connected_to_buffer = ""
            layer_buffer = parts[0].strip()
            output_shape_buffer = parts[1].strip()
            param_buffer = parts[2].strip()
            connected_to_buffer = parts[3].strip()
        else:
            if len(parts) == 1:
                connected_to_buffer += "" + stripped_line if connected_to_buffer.endswith((')')) or ']' in stripped_line else ""
                layer_buffer += "" + stripped_line if layer_buffer.endswith((']', ',')) or ')' in stripped_line else ""
            elif len(parts) == 2:
                connected_to_buffer += "" + parts[1] if connected_to_buffer.endswith((')')) or ']' in stripped_line else ""
                layer_buffer += "" + parts[0] if layer_buffer.endswith((']', ',')) or ')' in stripped_line else ""

        if (layer_buffer and output_shape_buffer and param_buffer and connected_to_buffer 
            and (layer_buffer.endswith(')') or layer_buffer.endswith(']'))
            and (connected_to_buffer.endswith(')') or connected_to_buffer.endswith(']'))
            and layer_buffer.count('(') == layer_buffer.count(')') and layer_buffer.count('[') == layer_buffer.count(']')
            and connected_to_buffer.count('(') == connected_to_buffer.count(')') and connected_to_buffer.count('[') == connected_to_buffer.count(']')):

            markdown_lines.append(f"| {layer_buffer:<28} | {output_shape_buffer:<28} | {param_buffer:<9} | {connected_to_buffer:<48} |")
            layer_buffer = ""
            output_shape_buffer = ""
            param_buffer = ""
            connected_to_buffer = ""

    if layer_buffer or output_shape_buffer or param_buffer or connected_to_buffer:
        markdown_lines.append(f"| {layer_buffer:<28} | {output_shape_buffer:<28} | {param_buffer:<9} | {connected_to_buffer:<48} |")

    markdown_lines.append("")
    markdown_lines.append(f"## Model parameters summary:")
    markdown_lines.append(f"**Paramter type** | **Count**")
    markdown_lines.append(f"|-|-|")
    markdown_lines.append(f"| **Total params:**           | **{total_params}**           |")
    markdown_lines.append(f"| **Trainable params:**       | **{trainable_params}**       |")
    markdown_lines.append(f"| **Non-trainable params:**   | **{non_trainable_params}**   |")

    # Join all lines into a single markdown string
    markdown_output = "\n".join(markdown_lines)
    return markdown_output

@st.dialog("Model Summary: model_1")
def summary_dialog(modelsummary):
    # Increase the width of the dialog, using dom for element with role "dialog"
    # st.markdown('<style>#root > div > div > div.st-dm {width: 90%}</style>', unsafe_allow_html=True) did not work. access properly through dom
    st.markdown('<style> div[role="dialog"] {width: 80%}</style>', unsafe_allow_html=True)
    summary = summary_to_markdown(modelsummary)
    st.markdown(summary)

st.title('Tweet Classification App')
st.write('This app classifies tweets into different categories based on their content and URL characteristics.')
tweet_text = st.text_area('Enter the tweet text:', 'Hackers exploit zero day bug. Link to the CVE: https://example.com/data')

col1, col2 = st.columns(2)
with col1:
    classify = st.button('Classify Tweet')
    if classify:
        urls_in_text = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_text)
        tweet_text_structure = np.array([extract_structural_features(tweet_text)])
        tweet_url_structure = extract_url_features(urls_in_text[0], urls_in_text, False)
        tweet_url_structure = url_hasher.transform([tweet_url_structure[-3:-1]]).toarray()

        tweet_text = remove_stopwords(tweet_text)
        tweet_text = replace_text_components(tweet_text)
        tweet_text = clean_text(tweet_text)
        tweet_text = lemmatize_text(tweet_text)
        tweet_text = tokenizer.texts_to_sequences([tweet_text])
        tweet_text = pad_sequences(tweet_text, maxlen=500)

        predictions = model.predict([tweet_text, tweet_text_structure, tweet_url_structure])
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_category = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]

        st.write(f'Tweet Category: {predicted_category} [Confidence: {predictions[0][predicted_label]:.2f}]')
        st.write('Confidence for other categories:')
        # Show pie chart with confidence for each category
        confidence_df = pd.DataFrame(predictions[0], columns=['Confidence'])
        confidence_df['Category'] = label_dict.keys()
        fig, ax = plt.subplots()
        ax.pie(confidence_df['Confidence'], labels=confidence_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        # with dialog, we can show the model summary on pressing the button

with col2:
    if st.button('Show Model Summary'):
        modelsummary = []
        model.summary(print_fn=lambda x: modelsummary.append(x))
        modelsummary = "\n".join(modelsummary)
        print(modelsummary)
        summary_dialog(modelsummary)

st.write('This app was created by me. You can find more about me on my [GitHub](https://www.github.com/ishan-surana).')