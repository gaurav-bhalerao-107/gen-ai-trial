# **Text Preprocessing and Tokenization in NLP**

## **Overview**  
This script performs **text preprocessing and tokenization**, which are essential steps in **Natural Language Processing (NLP)**. Preprocessing helps in **cleaning and standardizing text**, making it suitable for further analysis. The script includes **removing HTML tags, URLs, punctuations, stopwords, chat abbreviations expansion, spelling correction, tokenization, stemming, and lemmatization**.

---

## **Techniques Used and Why They Are Important**  

### **1. Lowercasing**
- **Used In**: `data = data.lower()`
- **Why?** Converting text to lowercase ensures **case insensitivity**, preventing duplicate representations (e.g., "AI" and "ai" should be treated the same).
- **Example**:
  ```python
  text = "Machine Learning is Amazing!"
  text = text.lower()
  print(text)
  ```
  **Output**:
  ```
  "machine learning is amazing!"
  ```
- ✅ **Advantages**: Standardizes text for comparison.  
- ❌ **Disadvantages**: Loses case-related emphasis (e.g., "Python" vs. "python").  
- **Use Cases**: Search engines, chatbots, NLP models.  

---

### **2. Removing HTML Tags**
- **Used In**: `remove_html_tags(data)`
- **Why?** Web-based texts often contain **HTML tags**, which do not contribute to NLP models.
- **Example**:
  ```python
  text = "<p>Hello, World!</p>"
  cleaned_text = remove_html_tags(text)
  print(cleaned_text)
  ```
  **Output**:
  ```
  "Hello, World!"
  ```
- ✅ **Advantages**: Removes unnecessary tags for clean text.  
- ❌ **Disadvantages**: May remove useful formatting.  
- **Use Cases**: Web scraping, social media analytics.  

---

### **3. Removing URLs**
- **Used In**: `remove_url(data)`
- **Why?** URLs are **not useful** for sentiment analysis or NLP tasks.
- **Example**:
  ```python
  text = "Check this out: https://example.com"
  cleaned_text = remove_url(text)
  print(cleaned_text)
  ```
  **Output**:
  ```
  "Check this out:"
  ```
- ✅ **Advantages**: Removes irrelevant text.  
- ❌ **Disadvantages**: May lose context if URL text contains information.  
- **Use Cases**: Sentiment analysis, spam filtering.  

---

### **4. Removing Punctuations**
- **Used In**: `remove_punctuations(data)`
- **Why?** Punctuations **do not add meaning** in NLP tasks.
- **Example**:
  ```python
  text = "Hello!!! How are you?"
  cleaned_text = remove_punctuations(text)
  print(cleaned_text)
  ```
  **Output**:
  ```
  "Hello How are you"
  ```
- ✅ **Advantages**: Standardizes text.  
- ❌ **Disadvantages**: May remove important meaning in some cases (e.g., sarcasm detection).  
- **Use Cases**: Speech-to-text, sentiment analysis.  

---

### **5. Expanding Chat Abbreviations**
- **Used In**: `chat_conversation(data)`
- **Why?** Chat messages and **social media texts use abbreviations** like "OMG" for "Oh My God".
- **Example**:
  ```python
  text = "OMG this is amazing!"
  expanded_text = chat_conversation(text)
  print(expanded_text)
  ```
  **Output**:
  ```
  "Oh My God this is amazing!"
  ```
- ✅ **Advantages**: Improves readability.  
- ❌ **Disadvantages**: Not all abbreviations are covered.  
- **Use Cases**: Chatbots, customer service automation.  

---

### **6. Spelling Correction**
- **Used In**: `handle_incorrect_text(data)`
- **Why?** Correcting spelling errors improves **text accuracy**.
- **Example**:
  ```python
  text = "Ths is an amzing model"
  corrected_text = handle_incorrect_text(text)
  print(corrected_text)
  ```
  **Output**:
  ```
  "This is an amazing model"
  ```
- ✅ **Advantages**: Increases text accuracy.  
- ❌ **Disadvantages**: Can incorrectly change proper nouns.  
- **Use Cases**: Autocorrect, grammar checking.  

---

### **7. Removing Stopwords**
- **Used In**: `remove_stopwords(data)`
- **Why?** Stopwords like "the", "is", and "and" **do not add meaning** to text analysis.
- **Example**:
  ```python
  text = "This is a good day"
  cleaned_text = remove_stopwords(text)
  print(cleaned_text)
  ```
  **Output**:
  ```
  "good day"
  ```
- ✅ **Advantages**: Reduces noise in text.  
- ❌ **Disadvantages**: Removing too many words may **lose context**.  
- **Use Cases**: Text summarization, search engines.  

---

### **8. Tokenization**
- **Used In**: `word_tokenization(data)`, `sentence_tokenization(data)`, `regex_word_tokenization(data)`, `spacy_word_tokenization(data)`, etc.
- **Why?** Splitting text into **words or sentences** allows NLP models to process it.
- **Example**:
  ```python
  text = "Hello world! AI is amazing."
  tokens = word_tokenization(text)
  print(tokens)
  ```
  **Output**:
  ```
  ["Hello", "world", "!", "AI", "is", "amazing", "."]
  ```
- ✅ **Advantages**: Splits text into meaningful parts.  
- ❌ **Disadvantages**: May **split incorrectly** for abbreviations.  
- **Use Cases**: Speech-to-text, machine translation.  

---

### **9. Stemming**
- **Used In**: `stem_words(data)`
- **Why?** Reduces words to their **root form** (e.g., "running" → "run").
- **Example**:
  ```python
  text = "running jumped"
  stemmed_text = stem_words(text)
  print(stemmed_text)
  ```
  **Output**:
  ```
  "run jump"
  ```
- ✅ **Advantages**: Reduces word variations.  
- ❌ **Disadvantages**: Stems may not be real words.  
- **Use Cases**: Search engines, topic modeling.  

---

### **10. Lemmatization**
- **Used In**: `lemmatize_words(data)`
- **Why?** Similar to stemming but **ensures words remain valid**.
- **Example**:
  ```python
  text = "running jumped"
  lemmatized_text = lemmatize_words(text)
  print(lemmatized_text)
  ```
  **Output**:
  ```
  "run jump"
  ```
- ✅ **Advantages**: More accurate than stemming.  
- ❌ **Disadvantages**: Slower processing speed.  
- **Use Cases**: Machine learning models, text summarization.  

---

## **Best Practices**
✅ Always **clean text before analysis** to improve accuracy.  
✅ Use **lemmatization over stemming** for meaningful words.  
✅ **Keep stopwords based on context** (e.g., **not** removing "not" in sentiment analysis).  
✅ **Use regex for efficient tokenization** in large text datasets.  
✅ Validate spelling correction to **avoid altering proper nouns**.  

---

## **Installation and Setup**  

### **1. Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies. Run the following commands:  

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **2. Install Dependencies**
Once inside the virtual environment, install the required dependencies using:  

```bash
pip install -r requirements.txt
```

### **3. Download Required NLP Models**
Some NLP libraries like **spaCy** require additional model downloads. Run the following command:  

```bash
python -m spacy download en_core_web_sm
```

### **4. Run**
Some NLP libraries like **spaCy** require additional model downloads. Run the following command:  

```bash
python app.py
```

---
