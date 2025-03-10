import re

# NLTK download
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# tokenization
from nltk.tokenize import word_tokenize,sent_tokenize

# stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# lemmatization
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

# word tokenization
def word_tokenization(text):
    return text.split()

# sentence tokenization
def sentence_tokenization(text):
    return text.split('.')

# regex word tokenization
def regex_word_tokenization(text):
    tokens = re.findall("[\w']+", text)
    return tokens

# regex sentence tokenization
def regex_sentence_tokenization(text):
    tokens = re.compile('[.!?] ').split(text)
    return tokens

# NLTK word tokenization
def nltk_word_tokenization(text):
    return word_tokenize(text)

# NLTK sentence tokenization
def nltk_sentence_tokenization(text):
    return sent_tokenize(text)


##### spacy #####
# spaCy is a fast, efficient, and powerful Natural Language Processing (NLP) library designed for 
# text analysis, tokenization, named entity recognition (NER), dependency parsing, and more. 
# It is widely used in AI and machine learning applications.
def spacy_word_tokenization(text):
    import spacy
    # RUN BELOW COMMAND - The en_core_web_sm model is downloaded separately by spaCy and stored outside of pip's package management system.
    
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        print("SpaCy model not found! Run the following command:")
        print("python -m spacy download en_core_web_sm")
        return ""
    
    nlp = spacy.load('en_core_web_sm')
    spacy_tokens = nlp(text)
    return spacy_tokens

##### stemming #####
# Stemming is a text-processing technique in Natural Language Processing (NLP) used to reduce words to their base form (stem/root word) by removing prefixes and suffixes. 
# The goal is to convert related words into the same root form, but it doesn't always guarantee meaningful words.
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

##### lemmatization #####
# Lemmatization is a text-processing technique used in Natural Language Processing (NLP) to reduce words to their base or root form (known as a lemma). 
# Unlike stemming, lemmatization considers the context and meaning of the word, ensuring that the transformed word remains a valid one.
def lemmatize_words(text):
    word_tokens = nltk_word_tokenization(text)
    tokens =  " ".join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in word_tokens])
    return tokens





