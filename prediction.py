import spacy
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


ro_stopwords = stopwords.words("romanian")
stemmer = SnowballStemmer("romanian")
nlp = spacy.load("ro_core_news_sm")


def read_data(filename):
    df = pd.read_json(filename)
    return df


def check_token(token):
    if token not in ro_stopwords and token not in list(
        string.punctuation) and len(token) > 2:
        return token


def clean_text(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ",text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ''.join(p for p in text if p not in string.punctuation)

    text = nlp(text)
    clean = [token.lemma_ for token in nlp(text) if check_token(token.text)]
    text = ""

    for i in clean:
        text = text + i + " "

    return text


def create_model(tfidf_train, rating_train):
    model = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        algorithm='brute',
        leaf_size='30',
        n_jobs=4
    )

    model.fit(tfidf_train, rating_train)
    return model


if __name__ == "__main__":
    train_data = read_data("train.json")
    test_data = read_data("test_wor.json")


    train_data['text'] = train_data['text'].apply(clean_text)
    train_data['text'].to_pickle("train_data.pkl")

    test_data['text'] = test_data['text'].apply(clean_text)
    test_data['text'].to_pickle("test_data.pkl")


    # load_model = spacy.load('ro', disable=['parser', 'ner'])
    # My_text = "This is just a sample text for the purpose of testing"
    # doc = load_model(My_text)
    # " ".join([token.lemma_ for token in doc])

    # nlp = spacy.load("ro_core_news_sm")
    # tt = train_data.iloc[1]
    # doc = nlp(tt['text'])
    # print(doc.text)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_, token.lemma_, token.norm_)

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_train = tfidf_vectorizer.fit_transform(train_data['text'])
    tfidf_test = tfidf_vectorizer.transform(test_data['text'])

    model = create_model(tfidf_train, train_data['rating'])

    p = model.predict(tfidf_test)

    prediction = pd.DataFrame({'text': test_data['text'], 'rating': p})

print("Ready")


