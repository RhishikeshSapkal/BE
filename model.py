import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("creditcard.csv")
x = dataset.iloc[:,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22]].values
y = dataset.iloc[:,-1].values


estimator = []
model1 = GaussianNB()
estimator.append(('NB',model1))
model2 = KNeighborsClassifier(n_neighbors=4, weights='uniform')
estimator.append(('KNN', model2))
model3 = RandomForestClassifier()
estimator.append(('RFC',model3))
ensemble = VotingClassifier(estimator, voting='hard' )
clf = ensemble.fit(x, y)

print("trained1")


pickle.dump(ensemble, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
