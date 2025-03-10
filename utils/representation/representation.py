import numpy as np
import pandas as pd
import os

# corpus
df = pd.DataFrame({
    "text":[
        "students read books",  
        "librarians read books",  
        "students write essays",  
        "librarians write essays"
    ],
    "output":[1,1,0,0]
})

# unique words in dataframe
sentences = [text for text in df['text']]
words = set([word for sentence in sentences for word in sentence.split()])
print("unique words in dataframe... ", words)    

# read data from multiple files
book = ""
path = "../../data/harry_potter_books"
for file_path in os.listdir(path):
    try:
        with open("{path}/{file_path}".format(path=path,file_path=file_path), 'r', encoding='utf-8') as file:
            book += file.read() + "\n" 
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred with file {file_path}: {e}")
print("length of book... ", len(book))

##### Bag Of Words #####
def bag_of_words(dataframe):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    
    cv_fit =  cv.fit_transform(dataframe['text'])

    # print vocab
    print("dataframe vocab... ", cv.vocabulary_)

    # print bag of words
    print("dataframe bag of words... ", cv_fit.toarray())

    # new text
    new_text = "librarians read students essays"
    new_text = cv.transform([new_text])

    # print vocab
    print("new text vocab... ", cv.vocabulary_)

    # print bag of words
    print("new text bag of words... ", new_text.toarray())

##### Bag Of Words With N-grams #####
def bag_of_words_with_ngrams(dataframe):
    from sklearn.feature_extraction.text import CountVectorizer
    
    # if ngram_range=(1,1) then bag of words
    # if ngram_range=(2,2) then bag of bigrams
    # if ngram_range=(3,3) then bag of trigrams
    cv = CountVectorizer(ngram_range=(2,2))

    cv_fit =  cv.fit_transform(dataframe['text'])

    # print vocab
    print("dataframe vocab... ", cv.vocabulary_)

    # print bag of words
    print("dataframe bag of words... ", cv_fit.toarray())


##### TF-IDF (Term frequency- Inverse document frequency) #####
def tf_idf(dataframe):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()

    tfidf_fit = tfidf.fit_transform(dataframe['text'])

    # print vocab
    print("dataframe vocab... ", tfidf.vocabulary_)

    # print bag of words
    print("dataframe bag of words... ", tfidf_fit.toarray())


##### Word2Vec #####
def word2vec(book):
    import gensim
    import nltk
    from nltk import sent_tokenize
    from gensim.utils import simple_preprocess
    nltk.download('punkt')
    import plotly.express as px

    story = []
    token = sent_tokenize(book)
    for sent in token:
        story.append(simple_preprocess(sent))
    print("story...", story)

    model = gensim.models.Word2Vec(
        window=10,
        min_count=2
    )

    model.build_vocab(story)

    model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

    harry = model.wv.most_similar('harry')
    print("similar harry...", harry)

    voldemort = model.wv.most_similar('voldemort')
    print("similar voldemort...", voldemort)

    harry_and_voldemort = model.wv.similarity('harry','voldemort')
    print("harry and voldemort...", harry_and_voldemort)

    vec = model.wv.get_normed_vectors()
    print("vectors...", vec)

    vec_shape = model.wv.get_normed_vectors().shape
    print("vectors shape...", vec_shape)

    index_to_key = model.wv.index_to_key


    # visualize word2vec
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X = pca.fit_transform(model.wv.get_normed_vectors())
    shape = X.shape
    print("pca shape...", shape)

    fig = px.scatter_3d(X[200:300],x=0,y=1,z=2, color=index_to_key[200:300])
    fig.show()



    
    

##### Bag Of Words #####
bag_of_words(df)

##### Bag Of Words With N-grams #####
bag_of_words_with_ngrams(df)

##### TF-IDF (Term frequency- Inverse document frequency) #####
tf_idf(df)

##### Word2Vec #####
word2vec(book)