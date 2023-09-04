import re
import nltk
import joblib
import pickle
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load the CountVectorizer and sentiment classifier
cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', 'rb'))
# classifier = joblib.load('c2_Classifier_Sentiment_Model.pkl')


import joblib
classifier = joblib.load('c2_Classifier_Sentiment_Model')

# Initialize the Porter Stemmer and stopwords
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def predict_sentiment(input_text):
    # Preprocess the input text
    review = re.sub('[^a-zA-Z]', ' ', input_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    # Transform the preprocessed text data into a BoW representation
    X_input = cv.transform([review]).toarray()

    # Predict sentiment label
    predicted_label = classifier.predict(X_input)[0]

    return predicted_label

# Example usage:
input_text  = 'Wow.. love this investment opportunity'
predicted_label = predict_sentiment(input_text)
print("Predicted Sentiment Label:", predicted_label)
