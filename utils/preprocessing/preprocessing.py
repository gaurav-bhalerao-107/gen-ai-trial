import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import emoji

nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.chat_words = {
            'AFAIK': 'As Far As I Know', 'AFK': 'Away From Keyboard', 'ASAP': 'As Soon As Possible',
            'FYI': 'For Your Information', 'BRB': 'Be Right Back', 'BTW': 'By The Way', 'OMG': 'Oh My God',
            'IMO': 'In My Opinion', 'LOL': 'Laugh Out Loud', 'TTYL': 'Talk To You Later', 'GTG': 'Got To Go',
            'TTYT': 'Talk To You Tomorrow', 'IDK': "I Don't Know", 'TMI': 'Too Much Information',
            'IMHO': 'In My Humble Opinion', 'ICYMI': 'In Case You Missed It', 'FAQ': 'Frequently Asked Questions',
            'TGIF': "Thank God It's Friday", 'FYA': 'For Your Action'
        }
    
    def remove_html_tags(self, text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)
    
    def remove_url(self, text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)
    
    def remove_punctuations(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def chat_conversation(self, text):
        new_words = [self.chat_words[word.upper()] if word.upper() in self.chat_words else word for word in text.split()]
        return ' '.join(new_words)
    
    def handle_incorrect_text(self, text):
        textBlb = TextBlob(text)
        return textBlb.correct().string
    
    def remove_stopwords(self, text):
        return " ".join([word for word in text.split() if word not in self.stop_words])
    
    def replace_emoji(self, text):
        return emoji.demojize(text)

# Example usage:
if __name__ == "__main__":
    text = """🚀 OMG! This AI model is AMAZING!!! 🤖🔥 I just read an article at https://ai-news.com about the future of AI. 
        BTW, did u see the new iPhone 15 Pro Max? It's sooo expensive! 😩💰 
        I'll BRB, need to grab some coffee ☕. Also, my friend said the weather is aweeesome today in California 😎🌞. 
        IDK why people still use old-school phones 📱😂. FYI, AI is gonna change everything ASAP!
    """
    preprocessor = TextPreprocessor()
    text = preprocessor.remove_html_tags(text)
    text = preprocessor.remove_url(text)
    text = preprocessor.replace_emoji(text)
    text = preprocessor.remove_punctuations(text)
    text = preprocessor.chat_conversation(text)
    print("Processed Text:", text)