# Sentiment Analysis Pipeline

## Overview
This project implements a **Sentiment Analysis Pipeline** using **Natural Language Processing (NLP) techniques**. It processes text data, extracts features, trains machine learning models, and evaluates their performance.

### **Key Features**
- **Data Preprocessing**: HTML tag removal, stopword removal, and text normalization.
- **Feature Extraction**: `Bag of Words (BoW)`, `TF-IDF`, and `Word2Vec`.
- **Classification Models**: `Naive Bayes` and `Random Forest`.
- **Performance Evaluation**: Accuracy and confusion matrix.

---

## **Techniques Used**
### **1. Data Preprocessing**
- **Removing HTML Tags**: Cleans unnecessary HTML elements.
- **Stopword Removal**: Removes common words like *"the"*, *"is"*, *"in"*, etc.
- **Lowercasing**: Converts text to lowercase to standardize data.

#### ✅ **Example**
```
Input: "<p>This movie was AMAZING!</p>"
After Preprocessing: "movie amazing"
```

---

### **2. Feature Extraction**
To convert text into numerical format, the following techniques are used:

#### **📌 Bag of Words (BoW)**
Represents text as a frequency matrix of words.
- Each document is a **vector** of word counts.
- **Ignores word order** and only captures word occurrence.

✅ **Example:**
```
Reviews: ["Great movie", "Terrible film", "Great film"]
BoW Representation:
    movie  |  film  |  great  |  terrible  
      1    |   0    |    1    |    0    
      0    |   1    |    0    |    1    
      0    |   1    |    1    |    0    
```

#### **📌 TF-IDF (Term Frequency - Inverse Document Frequency)**
- Assigns weight to words based on importance.
- **Common words** get lower weight, **unique words** get higher weight.

✅ **Example:**
```
Common words like "the", "is" have low weight.
Rare words like "fantastic" have high weight.
```

#### **📌 Word2Vec (Word Embeddings)**
- **Captures meaning and context** of words.
- Words with similar meaning have **similar vectors**.

✅ **Example:**
```
Word: "King" → [0.4, 0.7, -0.2, ...]
Word: "Queen" → [0.5, 0.6, -0.1, ...]
King and Queen are close in vector space!
```

---

### **3. Classification Models**
The following machine learning models are trained:

#### **📌 Naive Bayes**
- Based on **probability** of words appearing in each class.
- **Fast and efficient for text classification**.

✅ **Where to use?**
- Spam detection, sentiment analysis.

#### **📌 Random Forest**
- Ensemble of **multiple decision trees**.
- **Robust and accurate**, but slower than Naive Bayes.

✅ **Where to use?**
- When accuracy is more important than speed.

---

## **Advantages & Disadvantages**
| Technique | Advantages | Disadvantages |
|-----------|------------|--------------|
| **Bag of Words (BoW)** | Simple, fast | Ignores meaning & context |
| **TF-IDF** | Reduces impact of common words | Still ignores meaning |
| **Word2Vec** | Captures meaning | Needs large dataset |
| **Naive Bayes** | Fast & works well with text | Assumes word independence |
| **Random Forest** | High accuracy | Computationally expensive |

---

## **Best Practices**
✅ **Limit the number of features in `CountVectorizer`** (e.g., `max_features=2000`) to prevent memory overload.  
✅ **Use `sparse matrices`** instead of dense arrays to save memory.  
✅ **Use pre-trained embeddings (like `word2vec-google-news-300`)** instead of training Word2Vec on small datasets.  
✅ **Balance train-test split** to avoid bias (e.g., `test_size=0.2`).  
✅ **Optimize `RandomForestClassifier` parameters** to speed up training.  


## **Project Workflow**
1️⃣ **Data Loading & Cleaning**  
2️⃣ **Feature Extraction (BoW, TF-IDF, Word2Vec)**  
3️⃣ **Train Naive Bayes & Random Forest Classifiers**  
4️⃣ **Evaluate Model Performance**  

---

## **Results & Performance**
- **Naive Bayes with BoW Accuracy: ~80%**
- **Random Forest with TF-IDF Accuracy: ~85%**
- **Word2Vec with Random Forest Accuracy: ~88%**

---

## **Future Improvements**
🚀 **Try Deep Learning (LSTMs, Transformers like BERT)**  
🚀 **Optimize feature selection (reduce vocabulary size, use lemmatization)**  
🚀 **Use sentiment lexicons for better word representation**  

---

## **Conclusion**
This project provides a **modular sentiment analysis pipeline** that efficiently processes text, extracts features, and classifies sentiment using machine learning models. It follows **best practices in NLP** and can be extended for real-world applications.

---

### **Author**
👨‍💻 **Developed by:** Gaurav Bhalerao  
📧 **Contact:** gauravbhalerao107@gmail.com  
🌐 **GitHub:** [Your GitHub Profile]  

---

