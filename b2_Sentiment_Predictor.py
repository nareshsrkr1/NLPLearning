
import numpy as np
import pandas as pd
dataset = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv', delimiter = '\t', quoting = 3)
dataset.head()


# ###Data cleaning

# In[5]:


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[6]:


corpus=[]

for i in range(0, 100):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# ###Data transformation

# In[7]:


# Loading BoW dictionary
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile='c1_BoW_Sentiment_Model.pkl'
# cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
cv = pickle.load(open(cvFile, "rb"))


# In[8]:


X_fresh = cv.transform(corpus).toarray()
X_fresh.shape


# ###Predictions (via sentiment classifier)

# In[9]:


import joblib
classifier = joblib.load('c2_Classifier_Sentiment_Model')


# In[10]:


y_pred = classifier.predict(X_fresh)
print(y_pred)


# In[11]:


dataset['predicted_label'] = y_pred.tolist()
dataset.head()


# In[12]:


dataset.to_csv("c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)

