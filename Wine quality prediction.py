#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing important libraries

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[3]:


# import libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, r2_score


# ## EDA on white wine data

# In[4]:


winequality_white = pd.read_csv('winequality-white.csv', sep=';')       # load csv file
winequality_white


# In[5]:


winequality_white.info()


# In[6]:


winequality_white.isnull().sum().sum()


# In[7]:


winequality_white.columns


# In[8]:


winequality_white.describe()


# In[9]:


winequality_white.groupby('quality').mean()


# In[10]:


winequality_white.hist(figsize=(10,10),bins=50,grid=False)
plt.show()


# In[11]:


winequality_white['residual sugar'].skew()


# In[12]:


winequality_white['residual sugar'].plot(kind='box')


# In[13]:


winequality_white[winequality_white['residual sugar']>50]    # extreme outliers


# In[14]:


winequality_white.drop(index=[2781],inplace=True)    # drop outliers


# In[15]:


winequality_white['quality'].value_counts()   # check the range of diffrent wine quality


# In[16]:


sns.countplot(winequality_white, x='quality', palette='tab10')    # count plot of wine quality
plt.show()


# In[17]:


winequality_white.corr()


# In[18]:


plt.figure(figsize=(8,6))
sns.heatmap(winequality_white.corr(),annot=True,fmt='.1f', cmap='Blues')
plt.show()


# ### Feature selection

# In[19]:


winequality_white.shape


# In[20]:


# Convert into X and y to train the model
X = winequality_white.iloc[:,0:-1]
y = winequality_white.iloc[:,-1]


# In[21]:


print(X.shape)
print(y.shape)


# In[22]:


from imblearn.over_sampling import SMOTE
ros = SMOTE(k_neighbors=4, random_state=42)
X, y = ros.fit_resample(X, y)


# In[23]:


y.value_counts()


# In[24]:


# train test split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[25]:


X_train, X_val, y_train, y_val = train_test_split(X_train_temp,y_train_temp,test_size=0.2,random_state=42)


# In[26]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# In[27]:


# applying standard scalar

SC = StandardScaler()
xtrain_scaled = SC.fit_transform(X_train)
xval_scaled = SC.transform(X_val)
xtest_scaled = SC.transform(X_test)
xtrain_scaled = pd.DataFrame(xtrain_scaled,columns=X_train.columns)
xval_scaled = pd.DataFrame(xval_scaled,columns=X_val.columns)
xtest_scaled = pd.DataFrame(xtest_scaled,columns=X_test.columns)


# ## RandomForest Regressor

# In[28]:


# model selection and train the model
rf_white = RandomForestClassifier(oob_score=True)
rf_white.fit(xtrain_scaled,y_train)


# In[29]:


# prediction and accuracy
y_pred = rf_white.predict(xval_scaled)
print('Accuracy: {:0.2f}'.format(accuracy_score(y_val,y_pred)))
print('f1 score: {:0.2f}'.format(f1_score(y_val,y_pred, average='weighted')))


# In[30]:


rf_white.oob_score_


# ### Hyperparameter tuning

# In[31]:


rf_white_tuned = RandomForestClassifier(n_estimators=1000,oob_score=True)
rf_white_tuned.fit(xtrain_scaled,y_train)


# In[32]:


y_pred = rf_white_tuned.predict(xval_scaled)
print('Accuracy: {:0.2f}'.format(accuracy_score(y_val,y_pred)))
print('f1 score: {:0.2f}'.format(f1_score(y_val,y_pred, average='weighted')))


# In[33]:


rf_white_tuned.oob_score_


# In[34]:


# EVALUATION WITH TEST DATA

y_pred = rf_white_tuned.predict(xtest_scaled)
print('Accuracy: {:0.2f}'.format(accuracy_score(y_test,y_pred)))
print('f1 score: {:0.2f}'.format(f1_score(y_test,y_pred, average='weighted')))


# In[35]:


con_mat = confusion_matrix(y_test,y_pred)
print(con_mat)


# # Export model for web-app

# In[36]:


import pickle
pickle.dump(winequality_white,open('white_wine.pkl','wb'))
pickle.dump(rf_white_tuned,open('final_model.pkl','wb'))
pickle.dump(SC,open('preprocessing.pkl','wb'))

