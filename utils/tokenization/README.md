# **📌 Text Preprocessing for NLP**
This repository provides **text preprocessing functions** for **tokenization, stemming, and lemmatization** using **Python, NLTK, spaCy, and regex**. These functions help prepare text data for **Machine Learning, AI, and NLP applications**.

---

## **🚀 Features**
- **Tokenization**: Splitting text into words or sentences.
- **Stemming**: Reducing words to their root form.
- **Lemmatization**: Converting words to their dictionary base form.
- **Regex-based tokenization**: Splitting text using patterns.
- **spaCy tokenization**: Advanced NLP tokenization using deep learning models.

---

## **📌 Prerequisites**
### **1️⃣ Install Required Libraries**
```bash
pip install nltk spacy
```
### **2️⃣ Download Required NLP Models**
```bash
python -m spacy download en_core_web_sm
```

---

## **📂 Function Overview**
### **🔹 Tokenization (Breaking Text into Words or Sentences)**
**Tokenization** is the process of splitting text into **words (word tokenization)** or **sentences (sentence tokenization).**

#### **1. Word Tokenization**
```python
def word_tokenization(text):
    return text.split()
```
**Example:**
```python
word_tokenization("Hello world! NLP is awesome.")
```
**Output:**
```python
['Hello', 'world!', 'NLP', 'is', 'awesome.']
```

#### **2. NLTK Word Tokenization**
```python
from nltk.tokenize import word_tokenize
def nltk_word_tokenization(text):
    return word_tokenize(text)
```
**Example:**
```python
nltk_word_tokenization("I'm learning NLP, it's exciting!")
```
**Output:**
```python
['I', "'m", 'learning', 'NLP', ',', 'it', "'s", 'exciting', '!']
```

#### **3. Regex Word Tokenization**
```python
import re
def regex_word_tokenization(text):
    tokens = re.findall("[\w']+", text)
    return tokens
```
**Example:**
```python
regex_word_tokenization("I'm happy!")
```
**Output:**
```python
["I'm", "happy"]
```

#### **4. Sentence Tokenization (NLTK)**
```python
from nltk.tokenize import sent_tokenize
def nltk_sentence_tokenization(text):
    return sent_tokenize(text)
```
**Example:**
```python
nltk_sentence_tokenization("Hello world! How are you? NLP is fun.")
```
**Output:**
```python
['Hello world!', 'How are you?', 'NLP is fun.']
```

---

## **🔹 Stemming (Reducing Words to Their Root Form)**
**Stemming** reduces words to their **base form** but doesn’t always preserve meaning.

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])
```
**Example:**
```python
stem_words("running studies happily")
```
**Output:**
```python
'run studi happili'
```

---

## **🔹 Lemmatization (Finding Dictionary Root Forms)**
Lemmatization reduces words to their **actual dictionary root** while preserving meaning.

```python
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    word_tokens = nltk_word_tokenization(text)
    tokens = " ".join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in word_tokens])
    return tokens
```
**Example:**
```python
lemmatize_words("running studies happily")
```
**Output:**
```python
'run study happily'
```

---

## **🔹 spaCy Tokenization (Advanced NLP)**
```python
import spacy

def spacy_word_tokenization(text):
    nlp = spacy.load("en_core_web_sm")
    spacy_tokens = nlp(text)
    return [token.text for token in spacy_tokens]
```
**Example:**
```python
spacy_word_tokenization("I'm learning NLP, it's fun!")
```
**Output:**
```python
['I', "'m", 'learning', 'NLP', ',', 'it', "'s", 'fun', '!']
```

---

## **📌 Advantages & Disadvantages**
| Method | Advantages | Disadvantages |
|--------|------------|--------------|
| **Stemming** | Fast, lightweight | May not return real words |
| **Lemmatization** | Meaningful words, context-aware | Slower, requires POS tagging |
| **Regex Tokenization** | Simple, customizable | Doesn't handle complex language rules |
| **NLTK Tokenization** | Pre-trained, handles punctuation | Requires model downloads |
| **spaCy Tokenization** | Most accurate, deep learning-based | Requires large models |

---

## **📌 Best Practices**
✅ **Use NLTK for simple tokenization and stemming**  
✅ **Use spaCy for large-scale NLP projects**  
✅ **Use regex for custom tokenization rules**  
✅ **Use lemmatization over stemming for meaningful words**  

---

## **📌 Where to Use?**
- **Chatbots**: Tokenization for understanding messages.
- **Search Engines**: Stemming to match similar words.
- **AI & Machine Learning**: Lemmatization for feature engineering.

---

## **📌 Running the Script**
### **🔹 Clone the Repository**
```bash
git clone https://github.com/your-username/text-preprocessing-nlp.git
cd text-preprocessing-nlp
```
### **🔹 Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### **🔹 Run the Python Script**
```bash
python text_preprocessing.py
```

---

## **🚀 Conclusion**
This script provides a **comprehensive set of NLP preprocessing tools** for **tokenization, stemming, and lemmatization**. Choose the **right technique** based on **your project’s needs**! 🎯

🔹 **Found this useful? Give it a ⭐ on GitHub!**  

Let me know if you need any modifications! 😊🚀
