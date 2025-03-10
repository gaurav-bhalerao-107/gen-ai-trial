# README: Text Processing Techniques in NLP

## Overview
This script demonstrates various text processing techniques in Natural Language Processing (NLP) using Python. It covers:

1. **Bag of Words (BoW)**
2. **Bag of Words with N-grams**
3. **TF-IDF (Term Frequency-Inverse Document Frequency)**
4. **Word2Vec (Word Embeddings)**

These techniques help convert text into numerical representations, which are essential for machine learning models.

---

## Explanation of Techniques
### 1. Bag of Words (BoW)
BoW represents text as a frequency-based vector of words present in the dataset. Each unique word is assigned an index, and the text is converted into a vector of word counts.

#### Example:
Given two sentences:
- "Students read books"
- "Librarians write essays"

BoW converts them into vectors based on word occurrences:
| Word      | Students | Read | Books | Librarians | Write | Essays |
|-----------|---------|------|-------|------------|-------|--------|
| Sentence 1 | 1       | 1    | 1     | 0          | 0     | 0      |
| Sentence 2 | 0       | 0    | 0     | 1          | 1     | 1      |

#### Advantages:
- Simple to implement
- Works well for small datasets

#### Disadvantages:
- Ignores word meaning and order
- Results in sparse matrices (many zeros)

#### Best Use Cases:
- Text classification
- Spam detection

---

### 2. Bag of Words with N-grams
This technique extends BoW by considering sequences of `n` words (bigrams, trigrams, etc.), preserving some context.

#### Example:
For bigrams (`n=2`), the phrase "students read books" becomes:
- "students read"
- "read books"

#### Advantages:
- Captures some context
- Improves performance for structured text

#### Disadvantages:
- Increases dimensionality
- May require more data to be effective

#### Best Use Cases:
- Sentiment analysis
- Named entity recognition

---

### 3. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF measures word importance based on frequency in a document relative to other documents.

#### Formula:
\[ TF-IDF = TF \times IDF \]
Where:
- **TF (Term Frequency)** = Number of times a word appears in a document
- **IDF (Inverse Document Frequency)** = Logarithm of total documents divided by documents containing the word

#### Example:
If "books" appears frequently in a document but rarely in others, its TF-IDF score will be high.

#### Advantages:
- Reduces the impact of common words
- Useful for keyword extraction

#### Disadvantages:
- More complex than BoW
- Still ignores word order

#### Best Use Cases:
- Search engines
- Topic modeling

---

### 4. Word2Vec (Word Embeddings)
Word2Vec represents words as dense vectors in a multi-dimensional space, capturing semantic relationships.

#### Example:
In Word2Vec space:
- **Similar words**: "king" and "queen"
- **Word relationships**: `king - man + woman ≈ queen`

#### Advantages:
- Preserves word meaning and relationships
- Useful for deep learning models

#### Disadvantages:
- Requires large datasets
- Computationally expensive

#### Best Use Cases:
- Chatbots
- Recommendation systems

---

## Best Practices
- **Preprocess text**: Convert to lowercase, remove stopwords and punctuation
- **Choose the right model**: Use BoW for simple tasks, TF-IDF for feature selection, and Word2Vec for deep learning
- **Optimize performance**: Limit vocabulary size and use dimensionality reduction

