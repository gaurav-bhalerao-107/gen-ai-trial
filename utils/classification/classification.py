import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import gensim
from gensim.utils import simple_preprocess
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 255)


class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def remove_html_tags(self, text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    def remove_stopwords(self, text):
        new_text = [word for word in text.split() if word not in self.stop_words]
        return " ".join(new_text)

    def to_lowercase(self, text):
        return text.lower()

class DataFrameAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_shape(self):
        shape = self.df.shape
        print("Shape: ", shape)
        return shape

    def sentiment_count(self, column_name='sentiment'):
        if column_name in self.df.columns:
            count = self.df[column_name].value_counts()
            print("Sentiment count:", count)
            return count
        else:
            print(f"Column '{column_name}' not found.")
            return None

    def check_null_values(self):
        null_values = self.df.isnull().sum()
        print("Null values:", null_values)
        return null_values

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print("Duplicate count:", duplicates)
        return duplicates

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        print("Duplicates dropped.")
        return self.df
    



class SentimentAnalysisPipeline:
    def __init__(self, df, test_size=0.2, random_state=1):
        """Initialize with dataset and split parameters."""
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.encoder = LabelEncoder()
        self.cv = None  # CountVectorizer instance
        self.tfidf = None  # TfidfVectorizer instance
        self.word2vec_model = None  # Word2Vec Model
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.trained_model = None  # Store trained model
        self.feature_type = None  # Store feature extraction method


    def preprocess(self):
        self.X = self.df.iloc[:, 0:1]  # Select first column (reviews)
        self.Y = self.encoder.fit_transform(self.df['sentiment'])  # Encode sentiments
        print("Preprocessing complete...")

    def split_data(self):
        """Splits data into training and testing sets."""
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Training Shape: {self.X_train.shape}")
        print(f"Testing Shape: {self.X_test.shape}")

    def apply_bow(self, max_features=None, ngram_range=(1,1)):
        """Applies Bag of Words (BoW) feature extraction."""
        self.cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.X_train_bow = self.cv.fit_transform(self.X_train['review']).toarray()
        self.X_test_bow = self.cv.transform(self.X_test['review']).toarray()
        self.feature_type = "bow"
        print(f"Bag of Words applied with max_features={max_features}, ngram_range={ngram_range}")

    def apply_tfidf(self):
        """Applies TF-IDF feature extraction."""
        self.tfidf = TfidfVectorizer()
        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train['review']).toarray()
        self.X_test_tfidf = self.tfidf.transform(self.X_test['review']).toarray()
        self.feature_type = "tfidf"
        print("TF-IDF applied...")

    def apply_word2vec(self, vector_size=100, window=10, min_count=1, workers=4):
        """Applies Word2Vec embedding to convert words into vector representation."""
        review_sentences = [sent.split() for sent in self.df['review'].tolist()]
        
        self.word2vec_model = gensim.models.Word2Vec(
            sentences=review_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        # Save the model
        self.word2vec_model.save("../../models/MovieReviewAnalysis.model")
        
        # Train the model
        self.word2vec_model.train(
            review_sentences, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.epochs
        )
        
        print("Word2Vec model trained and saved...")

        # Convert text to word vectors
        X_vectors = []
        for words in review_sentences:
            word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            X_vectors.append(np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size))
        
        self.X = np.array(X_vectors)
        self.split_data()  # Re-split data with Word2Vec feature representation
        self.feature_type = "word2vec"

    def train_and_evaluate(self, model_type="random_forest", feature_type="bow"):
        """Trains and evaluates a classification model using selected feature extraction."""
        if feature_type == "bow":
            X_train_feature, X_test_feature = self.X_train_bow, self.X_test_bow
        elif feature_type == "tfidf":
            X_train_feature, X_test_feature = self.X_train_tfidf, self.X_test_tfidf
        elif feature_type == "word2vec":
            X_train_feature, X_test_feature = self.X_train, self.X_test
        else:
            raise ValueError("Invalid feature_type. Choose from 'bow', 'tfidf', or 'word2vec'.")

        # Choose classifier
        if model_type == "naive_bayes":
            self.trained_model = GaussianNB()
        elif model_type == "random_forest":
            self.trained_model = RandomForestClassifier()
        else:
            raise ValueError("Invalid model_type. Choose from 'naive_bayes' or 'random_forest'.")

        # Train and predict
        self.trained_model.fit(X_train_feature, self.Y_train)
        Y_pred = self.trained_model.predict(X_test_feature)

        # Evaluate model
        accuracy = accuracy_score(self.Y_test, Y_pred)
        confusion_mat = confusion_matrix(self.Y_test, Y_pred)

        print(f"{model_type.upper()} with {feature_type.upper()} Accuracy: {accuracy}")

    def predict_sentiment(self, text):
        preprocessor = Preprocessing()
        text = preprocessor.remove_html_tags(text)
        text = preprocessor.remove_stopwords(text)
        text = preprocessor.to_lowercase(text)

        if self.feature_type == "bow":
            text_vector = self.cv.transform([text]).toarray()
        elif self.feature_type == "tfidf":
            text_vector = self.tfidf.transform([text]).toarray()
        elif self.feature_type == "word2vec":
            words = text.split()
            word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            text_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.word2vec_model.vector_size)
            text_vector = text_vector.reshape(1, -1)
        else:
            raise ValueError("Feature extraction method not set.")

        prediction = self.trained_model.predict(text_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return sentiment



if __name__ == "__main__":
    data_path = "../../data/movie_reviews/IMDB_Dataset.csv"
    df = pd.read_csv(data_path)

    # new review
    new_review = "The movie was absolutely brilliant and fantastic!"

    # basic analysis
    df_analyzer = DataFrameAnalyzer(df)
    df_analyzer.get_shape()
    df_analyzer.sentiment_count()
    df_analyzer.check_null_values()
    df_analyzer.check_duplicates()
    df_analyzer.drop_duplicates()
    df_analyzer.check_duplicates()

    # basic processing
    preprocessor = Preprocessing()
    df['review'] = df['review'].apply(preprocessor.remove_html_tags)
    df['review'] = df['review'].apply(preprocessor.remove_stopwords)
    df['review'] = df['review'].apply(preprocessor.to_lowercase)

    # Create instance of SentimentAnalysisPipeline
    pipeline = SentimentAnalysisPipeline(df)

    # Step 1: Preprocessing and Splitting
    pipeline.preprocess()
    pipeline.split_data()

    # Step 2: Apply Feature Extraction & Train Models
    pipeline.apply_bow()  # Default BoW
    pipeline.train_and_evaluate(model_type="naive_bayes", feature_type="bow")  # Naive Bayes with BoW
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)

    pipeline.apply_bow()  # Default BoW
    pipeline.train_and_evaluate(model_type="random_forest", feature_type="bow")  # Random Forest with BoW
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)

    pipeline.apply_bow(max_features=3000)  # BoW with top 3000 words
    pipeline.train_and_evaluate(model_type="random_forest", feature_type="bow")  # Random Forest with BoW (3000)
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)

    pipeline.apply_bow(ngram_range=(2,2), max_features=5000)  # N-grams with BoW
    pipeline.train_and_evaluate(model_type="random_forest", feature_type="bow")  # Random Forest with N-grams
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)

    pipeline.apply_tfidf()  # TF-IDF feature extraction
    pipeline.train_and_evaluate(model_type="random_forest", feature_type="tfidf")  # Random Forest with TF-IDF
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)

    pipeline.apply_word2vec()  # Word2Vec feature extraction
    pipeline.train_and_evaluate(model_type="random_forest", feature_type="word2vec")  # Random Forest with Word2Vec
    predicted_sentiment = pipeline.predict_sentiment(new_review)
    print("Predicted Sentiment:", predicted_sentiment)