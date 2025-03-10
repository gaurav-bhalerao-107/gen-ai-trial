# Text Processing Pipeline

## Overview
This project implements a **Text Processing Pipeline** using **Natural Language Processing (NLP) techniques**. It processes text data, extracts unique words, reads multiple text files, applies feature extraction techniques like **Bag of Words (BoW), TF-IDF, and Word2Vec**, and visualizes word embeddings using PCA.

### **Key Features**
- **Extracting unique words**: Identifies unique words from a dataset.
- **Reading multiple text files**: Reads and processes multiple text files from a specified directory.
- **Feature Extraction**: Implements `Bag of Words (BoW)`, `N-Grams`, `TF-IDF`, and `Word2Vec`.
- **Word Embeddings & Similarity**: Uses Word2Vec to generate word embeddings and find similar words.
- **Dimensionality Reduction**: Applies PCA for 3D visualization of word vectors.

---

## **Techniques Used**

### **1. Unique Word Extraction**
Extracts unique words from a dataset by splitting text into words and storing only unique occurrences.

#### ✅ **Example:**
```
Input: ["students read books", "librarians read books"]
Unique Words: {'students', 'read', 'books', 'librarians'}
```

---

### **2. Bag of Words (BoW)**
BoW converts text into a numerical format by counting word occurrences.

#### ✅ **Example:**
```
Text Samples: ["students read books", "librarians read books"]
BoW Representation:
    students | read | books | librarians
      1     |  1   |   1   |     0
      0     |  1   |   1   |     1
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Simple & Fast | Ignores word order & meaning |
| Works well for small datasets | High-dimensional for large corpora |

---

### **3. Bag of Words with N-Grams**
N-Grams capture sequences of words instead of single words.
- **Unigrams (1-word sequences)**: Standard BoW.
- **Bigrams (2-word sequences)**: Captures word pairs.
- **Trigrams (3-word sequences)**: Captures word triplets.

#### ✅ **Example (Bigrams):**
```
Text: "students read books"
Bigrams: ["students read", "read books"]
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Preserves some word order | More computationally expensive |
| Improves performance on contextual data | Increases feature space size |

---

### **4. TF-IDF (Term Frequency - Inverse Document Frequency)**
TF-IDF assigns importance to words based on their frequency across multiple documents. 
- **Words that appear frequently in a document but rarely in others get a higher weight.**

#### ✅ **Example:**
```
Word: "books"
TF-IDF Score: Higher if it appears often in one document but rarely in others.
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Reduces importance of common words | Still ignores word order |
| Works well for document comparison | Requires more computation than BoW |

---

### **5. Word2Vec (Word Embeddings)**
Word2Vec converts words into **dense vector representations** where similar words have similar vector values. It preserves semantic meaning.

#### ✅ **Example:**
```
Word2Vec('harry') → [0.12, -0.55, 0.33, ...]
Word2Vec('voldemort') → [0.11, -0.54, 0.34, ...]
Similarity(harry, voldemort) → 0.89
```

| **Advantages** | **Disadvantages** |
|---------------|------------------|
| Captures word meanings & relationships | Needs large datasets |
| Useful for deep learning models | Computationally expensive |

---

### **6. PCA for Word Embedding Visualization**
PCA reduces the high-dimensional word embeddings into **3D space for visualization**.

#### ✅ **Example:**
```
PCA Transform: Converts 100-dimensional Word2Vec vectors into 3D space for visualization.
```

---

## **Best Practices**
✅ **Use BoW for simple text classification problems.**  
✅ **Use N-Grams when word order is important.**  
✅ **Use TF-IDF to filter out common words while keeping important words.**  
✅ **Use Word2Vec for deep learning applications and NLP models.**  
✅ **Use PCA to visualize high-dimensional embeddings.**  

---

## **Project Workflow**
1️⃣ **Load Dataset & Extract Unique Words**  
2️⃣ **Read & Process Text Files**  
3️⃣ **Apply Bag of Words & N-Grams**  
4️⃣ **Apply TF-IDF for Term Importance**  
5️⃣ **Generate Word2Vec Embeddings**  
6️⃣ **Visualize Word Embeddings with PCA**  

---

## **Conclusion**
This project provides a **comprehensive text processing pipeline** that extracts features from text, generates word embeddings, and visualizes them using PCA. The techniques used are essential for **text classification, search engines, chatbots, and NLP applications.**

---

### **Author**
👨‍💻 **Developed by:** Gaurav Bhalerao  
📧 **Contact:** gauravbhalerao107@gmail.com  
🌐 **GitHub:** https://github.com/gaurav-bhalerao-107 

---