#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[27]:


import re
import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


# # LOAD DATASET

# In[2]:


df = pd.read_csv(r"C:\Users\jbaby\documents\DLK\personality\mbti_1.csv")


# # DATA - PREPROCESSING

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['type'].value_counts()


# # EXTRACTING FEATURES 

# In[8]:


df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

df.head()


# In[9]:


df.dtypes


# In[10]:


df.columns


# # PERSONALITY TYPE 
# 

# <span style="color: orange;font-size: 20px;">Personality type across 4 axis:
# </span>
# 
# Introversion (I) – Extroversion (E)
# 
# Intuition (N) – Sensing (S)
# 
# Thinking (T) – Feeling (F)
# 
# Judging (J) – Perceiving (P)
# 

# In[11]:


map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)
df.head()


# # VISUALIZATION

# In[12]:


plt.figure(figsize=(15,10))
sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.show()


# In[13]:


plt.figure(figsize=(15,10))
sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df, kind='kde')


# In[14]:


i = df['type'].unique()
k = 0
for m in range(0,2):
    for n in range(0,6):
        df_2 = df[df['type'] == i[k]]
        sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df_2, kind="hex")
        plt.title(i[k])
        k+=1


# In[15]:


i = df['type'].unique()
k = 0
TypeArray = []
PearArray=[]
for m in range(0,2):
    for n in range(0,6):
        df_2 = df[df['type'] == i[k]]
        pearsoncoef1=np.corrcoef(x=df_2['words_per_comment'], y=df_2['ellipsis_per_comment'])
        pear=pearsoncoef1[1][0]
        TypeArray.append(i[k])
        PearArray.append(pear)
        k+=1


TypeArray = [x for _,x in sorted(zip(PearArray,TypeArray))]
PearArray = sorted(PearArray, reverse=True)
plt.scatter(TypeArray, PearArray)


# In[16]:


print(i)


# In[17]:


words_per_comment_values = df['words_per_comment']

# Plot a histogram
plt.hist(words_per_comment_values, bins=20, edgecolor='black')
plt.title('Distribution of Words per Comment')
plt.xlabel('Words per Comment')
plt.ylabel('Frequency')
plt.show()


# # SPLITING DATASET 

# In[18]:


X = df.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
y = df['type'].values


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=5)


# # MODEL IMPLEMENTATION

# In[20]:


#Stochastic Gradient Descent (SGD)
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, y_train)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print(round(acc_sgd,2,), "%")


# In[21]:


# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[22]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")


# In[23]:


# K-Nearest Neighbor (KNN)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(round(acc_knn,2,), "%")


# In[24]:


# Support Vector Classifier (SVC)
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

acc_svm = round(svm_classifier.score(X_train, y_train) * 100, 2)
print(round(acc_svm,2,), "%")


# # COLLECT USER INPUT

# In[25]:


columns = ['words_per_comment', 'http_per_comment', 'music_per_comment', 
           'question_per_comment', 'img_per_comment', 'excl_per_comment', 
           'ellipsis_per_comment']

input_values = {}
for column in columns:
    value = input(f"Enter value for {column}: ")
    input_values[column] = float(value)  # Assuming the values are numeric, adjust data type if needed
input_values


# # MODEL PREDICTION

# In[26]:


prediction = random_forest.predict([list(input_values.values())])
print("Predicted Output:", prediction[0])


# In[29]:


with open('trained_model.pkl', 'wb') as file:
    pickle.dump(prediction, file)


# In[ ]:




