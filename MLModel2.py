#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df= pd.read_csv('train.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'no title'
    

def shorter_title(x):
    title=x['Title']
    if title in['Capt','Col','Major']:
        return 'Officer'
    elif title in['Jonkheer','Don','Lady','Master','the Countess','Rev']:
        return 'Royalty'
    elif title in['Miss','Mlle','Mme','Mrs','Ms']:
        return "Miss"
    elif title in ['Sir','Mr']:
        return 'Mr'
    else:
        return  title

df['Title']=df['Name'].map(lambda x: get_title(x))
df['Title']=df.apply(shorter_title, axis=1)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df.Sex.replace(('male','female'),(0,1), inplace=True)
df.Embarked.replace(('S','C','Q'),(0,1,2), inplace=True)
df.Title.replace(('Mr','Miss','Royalty','Dr','Officer'),(0,1,2,3,4), inplace=True)

y=df['Survived']
x=df.drop(['Survived', 'PassengerId'], axis=1)

xtrain, xval, ytrain, yval=train_test_split(x,y,test_size=0.1)

randomforest=RandomForestClassifier()
randomforest.fit(xtrain.values, ytrain.values)
pickle.dump(randomforest,open('titanic_model.sav','wb'))


