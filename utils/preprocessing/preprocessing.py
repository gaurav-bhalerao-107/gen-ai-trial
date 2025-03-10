import re

# NLTK stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# remove html tags
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

# remove url
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# remove punctuations
def remove_punctuations(text):
    import string,time
    punctuations = string.punctuation

    return text.translate(str.maketrans('', '', punctuations))


# chat conversation
def chat_conversation(text):
    chat_words = {
        'AFAIK':'As Far As I Know',
        'AFK':'Away From Keyboard',
        'ASAP':'As Soon As Possible',
        "FYI": "For Your Information",
        "ASAP": "As Soon As Possible",
        "BRB": "Be Right Back",
        "BTW": "By The Way",
        "OMG": "Oh My God",
        "IMO": "In My Opinion",
        "LOL": "Laugh Out Loud",
        "TTYL": "Talk To You Later",
        "GTG": "Got To Go",
        "TTYT": "Talk To You Tomorrow",
        "IDK": "I Don't Know",
        "TMI": "Too Much Information",
        "IMHO": "In My Humble Opinion",
        "ICYMI": "In Case You Missed It",
        "AFAIK": "As Far As I Know",
        "BTW": "By The Way",
        "FAQ": "Frequently Asked Questions",
        "TGIF": "Thank God It's Friday",
        "FYA": "For Your Action",
        "ICYMI": "In Case You Missed It",
    }

    new_words = []
    for word in text.split():
        if word.upper() in chat_words:
            new_words.append(chat_words[word.upper()])
        else:
            new_words.append(word)
    return ' '.join(new_words)


# incorrect text handling
def handle_incorrect_text(text):
    from textblob import TextBlob

    textBlb = TextBlob(text)
    correct = textBlb.correct().string
    return correct


def remove_stopwords(text):
    new_text = []

    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)


def replace_emoji(text):
    import emoji
    
    return emoji.demojize(text)