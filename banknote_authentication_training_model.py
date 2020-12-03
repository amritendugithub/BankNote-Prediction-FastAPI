# Youtube Reference: https://youtu.be/b5F667g1yCk
# Dataset - https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data?select=BankNote_Authentication.csv

import numpy as np
import pandas as pd

df = pd.read_csv("BankNote_Authentication.csv")
#print(df.head())

# Independent Features, Dependept Features

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#print(y.head())

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 0)

# Implement RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Check accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)

print("Accuracy score {accuracy_score}".format(accuracy_score=score))

# Create a pickle file using serialization of the Classifier Model
import pickle
pickle_out = open("classifier.pickle","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()

