#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt


# In[2]:


from nlppreprocess import NLP


# In[3]:


nlp=NLP()


# In[4]:


from textblob import TextBlob


# In[5]:


df=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\sentiment_tweets3.csv")


# In[6]:


df.head()


# In[7]:


def cleantxt(text):
    text=re.sub(r'@[A-za_z0-9]+'," ",text)
    text=re.sub(r'#'"!",' ',text)
    text=re.sub(r'RT[\S]+',' ',text)
    text=re.sub(r'http?:\/\/\S+',' ',text)
    return text


# In[8]:


df["message to examine"]=df["message to examine"].apply(cleantxt)


# In[9]:


df["message to examine"]=df["message to examine"].apply(nlp.process)


# In[10]:


df.head()


# In[11]:


def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# In[12]:


def getpolarity(text):
    return TextBlob(text).sentiment.polarity


# In[13]:


df["Subjectivity"]=df["message to examine"].apply(getsubjectivity)
df["Polarity"]=df["message to examine"].apply(getpolarity)


# In[14]:


df.head()


# In[15]:


df.tail()


# In[16]:


def getAnalysis(score):
    if score<0:
        return "negative"
    elif score==0:
        return "netural"
    else:
        return "positive"


# In[17]:


df["Analysis"]=df["Polarity"].apply(getAnalysis)


# In[18]:


df.head()


# In[19]:


df["Analysis"].value_counts()
plt.title("Sentiment analysis")
plt.xlabel("Sentiment")
plt.ylabel("counts")
df["Analysis"].value_counts().plot(kind="bar")
plt.show()


# In[20]:


df=df[df["Polarity"]!=0]


# In[21]:


df["Positive Rated"]=np.where(df["Polarity"]<0,0,1)


# In[22]:


df["Positive Rated"].value_counts()


# In[23]:


df.head()


# In[24]:


import seaborn as sns


# In[25]:


sns.countplot(df["Positive Rated"])


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(df["message to examine"],df["Positive Rated"],random_state=50)


# In[28]:


print(x_train)


# In[29]:


print(y_train)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[31]:


vect=TfidfVectorizer().fit(x_train)


# In[32]:


len(vect.get_feature_names())


# In[33]:


x_train_vectorized=vect.transform(x_train)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model=LogisticRegression()


# In[36]:


model.fit(x_train_vectorized,y_train)


# In[37]:


pred=model.predict(vect.transform(x_test))


# In[38]:


print(pred)


# In[39]:


from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report


# In[40]:


print("AUC:",roc_auc_score(y_test,pred))


# In[41]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))


# In[42]:


feature_names=np.array(vect.get_feature_names())


# In[43]:


sorted_coef_index = model.coef_[0].argsort()


# In[44]:


print("Smallest coef", feature_names[sorted_coef_index[:10]])


# In[45]:


print("Largest coef", feature_names[sorted_coef_index[:-11:-1]])


# In[ ]:




