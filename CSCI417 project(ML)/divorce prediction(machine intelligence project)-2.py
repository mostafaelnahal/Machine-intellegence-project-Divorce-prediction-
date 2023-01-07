#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = df=pd.read_excel('Downloads/divorce/divorce.xlsx')


# In[3]:


df.head()


# In[4]:


#(0=Never, 1=Seldom, 2=Averagely, 3=Frequently, 4=Always).
df.info()


# In[5]:


import seaborn as sns

sns.pairplot(df, hue='Class')


# In[38]:


df.Class.value_counts().plot(kind='bar')


# In[6]:


y=df.loc[:,"Class"]
x=df.drop(["Class"],axis=1)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler #feature scalling (normalization) always scale trainning before test
sc=StandardScaler()
trainning_scaled=sc.fit_transform(X_train)
test_scaled=sc.transform(X_test)


# In[9]:


from sklearn.svm import SVC
svmmodel=SVC(kernel='poly', random_state=0)
svmmodel.fit(X_train,y_train)


# In[10]:


y_predict_train=svmmodel.predict(X_train)
y_predict_test=svmmodel.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[12]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[13]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[14]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)


# In[18]:


y_predict_train=DTC.predict(X_train)
y_predict_test=DTC.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[19]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[20]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier()
RF.fit(X_train,y_train)


# In[22]:


y_predict_train=RF.predict(X_train)
y_predict_test=RF.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[23]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[24]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)


# In[27]:


y_predict_train=KNN.predict(X_train)
y_predict_test=KNN.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[28]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[29]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[30]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,y_train)


# In[31]:


y_predict_train=NB.predict(X_train)
y_predict_test=NB.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[32]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[33]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[34]:


from sklearn.linear_model import LogisticRegression
LG = LogisticRegression(random_state=0)
LG.fit(X_train,y_train)


# In[35]:


y_predict_train=LG.predict(X_train)
y_predict_test=LG.predict(X_test)
from sklearn.metrics import accuracy_score
print("trainning acc= ",accuracy_score(y_train,y_predict_train))
print("test acc= ",accuracy_score(y_test,y_predict_test))


# In[36]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predict_test)
print(cn)


# In[37]:


from sklearn.metrics import f1_score 
f1_score(y_test,y_predict_test)


# In[ ]:




