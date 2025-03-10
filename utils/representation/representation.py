import os
import numpy as np
import pandas as pd
import gensim
import nltk
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px

nltk.download('punkt')

class TextProcessing:
    def __init__(self, dataframe, data_path):
        self.df = dataframe
        self.data_path = data_path
        self.book = ""

    def get_unique_words(self):
        sentences = [text for text in self.df['text']]
        words = set([word for sentence in sentences for word in sentence.split()])
        print("Unique words in dataframe:", words)
        return words

    def read_multiple_files(self):
        for file_path in os.listdir(self.data_path):
            try:
                with open(f"{self.data_path}/{file_path}", 'r', encoding='utf-8') as file:
                    self.book += file.read() + "\n"
            except FileNotFoundError:
                print(f"File {file_path} not found.")
            except Exception as e:
                print(f"An error occurred with file {file_path}: {e}")
        print("Length of book:", len(self.book))

    def bag_of_words(self):
        cv = CountVectorizer()
        cv_fit = cv.fit_transform(self.df['text'])
        print("Dataframe vocab:", cv.vocabulary_)
        print("Dataframe bag of words:", cv_fit.toarray())

    def bag_of_words_with_ngrams(self, ngram_range=(2,2)):
        cv = CountVectorizer(ngram_range=ngram_range)
        cv_fit = cv.fit_transform(self.df['text'])
        print("Dataframe vocab:", cv.vocabulary_)
        print("Dataframe bag of words:", cv_fit.toarray())

    def tf_idf(self):
        tfidf = TfidfVectorizer()
        tfidf_fit = tfidf.fit_transform(self.df['text'])
        print("Dataframe vocab:", tfidf.vocabulary_)
        print("Dataframe TF-IDF values:", tfidf_fit.toarray())

    def word2vec(self):
        story = []
        token = sent_tokenize(self.book)
        for sent in token:
            story.append(simple_preprocess(sent))
        print("Story:", story)

        model = gensim.models.Word2Vec(window=10, min_count=2)
        model.build_vocab(story)
        model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

        print("Similar to 'harry':", model.wv.most_similar('harry'))
        print("Similar to 'voldemort':", model.wv.most_similar('voldemort'))
        print("Similarity between 'harry' and 'voldemort':", model.wv.similarity('harry', 'voldemort'))

        vec = model.wv.get_normed_vectors()
        print("Vectors:", vec)
        print("Vector shape:", vec.shape)

        index_to_key = model.wv.index_to_key

        pca = PCA(n_components=3)
        X = pca.fit_transform(vec)
        print("PCA shape:", X.shape)

        fig = px.scatter_3d(X[200:300], x=0, y=1, z=2, color=index_to_key[200:300])
        fig.show()

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "text": [
            "students read books",  
            "librarians read books",  
            "students write essays",  
            "librarians write essays"
        ],
        "output": [1, 1, 0, 0]
    })
    
    processor = TextProcessing(df, "../../data/harry_potter_books")
    processor.get_unique_words()
    processor.read_multiple_files()
    processor.bag_of_words()
    processor.bag_of_words_with_ngrams()
    processor.tf_idf()
    processor.word2vec()