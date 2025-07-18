import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')

print(data.shape)
print(target.shape,target)

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

print(train_data.shape,train_target.shape)
print(test_data.shape,test_target.shape)

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier() #load KNN algorithm into model

model.fit(train_data,train_target) #training the KNN model using traininig data and target

predicted_target=model.predict(test_data) #getting predictions from the model
print(predicted_target)

print(test_target)

from sklearn.metrics import accuracy_score

acc=accuracy_score(test_target,predicted_target)
print('Accuracy:',acc)

from sklearn.metrics import classification_report

classi_report=classification_report(test_target,predicted_target)
print('Classification Report:',classi_report)

import joblib

joblib.dump(model,'sinhala-character-knn.sav')