import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextTokenizer:
    def __init__(self):
        self.ps = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("SpaCy model not found! Run the following command:")
            print("python -m spacy download en_core_web_sm")
            self.nlp = None

    def word_tokenization(self, text):
        return text.split()

    def sentence_tokenization(self, text):
        return text.split('.')

    def regex_word_tokenization(self, text):
        tokens = re.findall("[\w']+", text)
        return tokens

    def regex_sentence_tokenization(self, text):
        tokens = re.compile('[.!?] ').split(text)
        return tokens

    def nltk_word_tokenization(self, text):
        return word_tokenize(text)

    def nltk_sentence_tokenization(self, text):
        return sent_tokenize(text)

    def spacy_word_tokenization(self, text):
        if self.nlp:
            return [token.text for token in self.nlp(text)]
        return []

    def stem_words(self, text):
        return " ".join([self.ps.stem(word) for word in text.split()])

    def lemmatize_words(self, text):
        word_tokens = self.nltk_word_tokenization(text)
        tokens = " ".join([self.wordnet_lemmatizer.lemmatize(word, pos='v') for word in word_tokens])
        return tokens

# Example usage
if __name__ == "__main__":
    text = "The students are studying hard for their upcoming exams. Teachers are teaching in the classrooms."
    tokenizer = TextTokenizer()
    
    print("Word Tokenization:", tokenizer.word_tokenization(text))
    print("Sentence Tokenization:", tokenizer.sentence_tokenization(text))
    print("Regex Word Tokenization:", tokenizer.regex_word_tokenization(text))
    print("Regex Sentence Tokenization:", tokenizer.regex_sentence_tokenization(text))
    print("NLTK Word Tokenization:", tokenizer.nltk_word_tokenization(text))
    print("NLTK Sentence Tokenization:", tokenizer.nltk_sentence_tokenization(text))
    print("Spacy Word Tokenization:", tokenizer.spacy_word_tokenization(text))
    print("Stemming:", tokenizer.stem_words(text))
    print("Lemmatization:", tokenizer.lemmatize_words(text))
