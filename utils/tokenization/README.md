# Text Tokenization and Preprocessing

## Overview
This project implements a **Text Tokenization and Preprocessing Pipeline** using **Natural Language Processing (NLP) techniques**. It processes text by breaking it down into tokens (words and sentences), applies stemming and lemmatization, and provides multiple methods for text tokenization, including **regex-based, NLTK-based, and spaCy-based approaches**.

### **Key Features**
- **Tokenization**: Splitting text into words and sentences using different techniques.
- **Stemming**: Reducing words to their root form using Porter Stemmer.
- **Lemmatization**: Converting words to their dictionary base form.
- **Support for Multiple Libraries**: Uses **NLTK, Regex, and spaCy** for NLP tasks.

---

## **Techniques Used**

### **1. Tokenization**
Tokenization is the process of **splitting text into smaller units** (words or sentences). It helps in preparing text data for further analysis.

#### ✅ **Example:**
```
Input: "The students are studying hard for their exams. Teachers are teaching."

Word Tokenization:
['The', 'students', 'are', 'studying', 'hard', 'for', 'their', 'exams.', 'Teachers', 'are', 'teaching.']

Sentence Tokenization:
['The students are studying hard for their exams.', 'Teachers are teaching.']
```

#### **Types of Tokenization in the Code:**
| Tokenization Method | Library Used |
|---------------------|--------------|
| **Word Tokenization (split method)** | Python String Split |
| **Sentence Tokenization (split method)** | Python String Split |
| **Regex Word Tokenization** | Regex (`re.findall`) |
| **Regex Sentence Tokenization** | Regex (`re.compile`) |
| **NLTK Word Tokenization** | `nltk.word_tokenize()` |
| **NLTK Sentence Tokenization** | `nltk.sent_tokenize()` |
| **spaCy Word Tokenization** | `spacy.load('en_core_web_sm')` |


---

### **2. Stemming**
Stemming reduces words to their root form by **removing prefixes and suffixes**.

#### ✅ **Example:**
```
Input: "studying studies studied"
After Stemming: "study studi studi"
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Fast and efficient | May generate non-meaningful words |
| Works well for basic NLP tasks | Loses contextual meaning |

---

### **3. Lemmatization**
Lemmatization converts words to their **dictionary base form** by considering the context and meaning of the word.

#### ✅ **Example:**
```
Input: "studying studies studied"
After Lemmatization: "study study study"
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Provides valid words | Slightly slower than stemming |
| Contextually accurate | Requires additional NLP resources |

---

## **Best Practices**
✅ Use **lemmatization instead of stemming** when accuracy is more important than speed.  
✅ Use **spaCy for large-scale NLP tasks**, as it is optimized for performance.  
✅ Choose **Regex tokenization** when you need full control over splitting patterns.  
✅ Use **NLTK tokenization** for compatibility with classical NLP techniques.  


## **Project Workflow**
1️⃣ **Text Tokenization (Word & Sentence)**  
2️⃣ **Apply Stemming**  
3️⃣ **Apply Lemmatization**  
4️⃣ **Compare Results from Different Tokenization Methods**  

---

## **Conclusion**
This project provides a **modular NLP pipeline** that efficiently processes text by tokenizing, stemming, and lemmatizing it. The use of multiple techniques ensures flexibility and adaptability for real-world applications.

---

### **Author**
👨‍💻 **Developed by:** Gaurav Bhalerao
📧 **Contact:** gauravbhalerao107@gmail.com  
🌐 **GitHub:** https://github.com/gaurav-bhalerao-107

---