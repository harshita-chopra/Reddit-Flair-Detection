# Text preprocessing functions

import string
import re
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

file_content = open("stop_hinglish.txt").read()
STOPWORDS = word_tokenize(file_content)
punct = list(string.punctuation)
punct += '’'
lemmatizer = WordNetLemmatizer()

def remove_URL(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuations(text):
    for punctuation in punct:
        text = text.replace(punctuation, '')
    return text

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_digits(text):
    return re.sub(r"\d", "", text)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_text(sentence):
    # Tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

# To clean text in the passed column of the data by combining the above functions

def clean_text(df, col):
    print('Cleaning text of',col,'...')
    # Converting to lower case
    df[col]= df[col].str.lower()
    # Removing /n characters
    df[col]= df[col].apply(lambda x: x.replace('\n', ' '))
    # Removing urls
    df[col]= df[col].apply(lambda text: remove_URL(text))
    # Removing punctuations
    df[col]= df[col].apply(lambda text: remove_punctuations(text))
    # Removing the stopwords
    df[col]= df[col].apply(lambda text: remove_stopwords(text))
    # Remove the digits
    df[col]= df[col].apply(lambda text: remove_digits(text))
    # Lemmatization of text
    df[col]= df[col].apply(lambda text: lemmatize_text(text))
    print('DONE! \n')
