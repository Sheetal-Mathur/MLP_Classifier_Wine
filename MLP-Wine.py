import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
"Loading the dataset"
dataset = pd.read_csv('C:/Users/mathu/Downloads/winequality.csv')
print(dataset.head())
print(dataset.info())
"Here we are spliting the dataset into train test "
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.loc[:, dataset.columns != 'quality'], dataset['quality'],stratify=dataset['quality'],random_state=100)
print(dataset.loc[:,dataset.columns !='quality'])
print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train_transform=scaler.fit_transform(x_train)
x_test_tansform=scaler.fit_transform(x_test)
print(x_train_transform[0:10])


from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(300,150,100,45),activation='relu',max_iter=300,solver='adam')
model.fit(x_train_transform,y_train)

ypred=model.predict(x_test_tansform)
print(ypred)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy =accuracy_score(y_test,ypred)
print('accuracy is',accuracy)


cm=confusion_matrix(y_test,ypred)
print(cm)

from sklearn.metrics import classification_report
cr=classification_report(y_test,ypred)
print(cr)

plt.plot(model.loss_curve_)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()


#Applying grid and cross validation
max_iter=[100,200,250,150,300]

solver=['sgd','adam']

activation=['tanh','sigmoid','relu']

alpha=[0.0001,0.10,0.05]

from sklearn.model_selection import GridSearchCV

param_grid={'max_iter':max_iter,'solver':solver,'activation':activation,'alpha':alpha}

gridsearch=GridSearchCV(model,param_grid,cv=5)

gridsearch.fit(x_train,y_train)
