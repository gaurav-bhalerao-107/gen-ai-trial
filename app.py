from utils.preprocessing import *
from utils.tokenization import *

# Sample Data (Real-World Text)
data = """🚀 OMG! This AI model is AMAZING!!! 🤖🔥 I just read an article at https://ai-news.com about the future of AI. 
BTW, did u see the new iPhone 15 Pro Max? It's sooo expensive! 😩💰 
I'll BRB, need to grab some coffee ☕. Also, my friend said the weather is aweeesome today in California 😎🌞. 
IDK why people still use old-school phones 📱😂. FYI, AI is gonna change everything ASAP!
"""

# Preprocessing Steps
data = data.lower()
data = remove_html_tags(data)
data = remove_url(data)
data = remove_punctuations(data)
data = chat_conversation(data)
data = handle_incorrect_text(data)
data = remove_stopwords(data)
data = replace_emoji(data)

# Tokenization
word_tokens = word_tokenization(data)
sentence_tokens = sentence_tokenization(data)
regex_word_tokens = regex_word_tokenization(data)
regex_sentence_tokens = regex_sentence_tokenization(data)
spacy_word_tokens = spacy_word_tokenization(data)
nltk_word_tokens = nltk_word_tokenization(data)
nltk_sentence_tokens = nltk_sentence_tokenization(data)

# Stemming & Lemmatization
stemmed_text = stem_words(data)
lemmatized_text = lemmatize_words(data)

# Printing Results
print("Word Tokenization:", word_tokens)
print("Sentence Tokenization:", sentence_tokens)
print("Regex Word Tokenization:", regex_word_tokens)
print("Regex Sentence Tokenization:", regex_sentence_tokens)
print("SpaCy Word Tokenization:", spacy_word_tokens)
print("NLTK Word Tokenization:", nltk_word_tokens)
print("NLTK Sentence Tokenization:", nltk_sentence_tokens)
print("Stemming:", stemmed_text)
print("Lemmatization:", lemmatized_text)
