import pandas as pd

dataset = pd.read_csv(r'D:\ML day 5\Day1\iris.csv').values

print (dataset,dataset.shape)

data = dataset[:,0:4]
print(data,data.shape)

target = dataset[:,4]
print(target,target.shape)

from sklearn.model_selection import train_test_split

train_data , test_data , train_target , test_target = train_test_split(data,target,test_size=0.2)

print (test_data,test_data.shape)
print (test_target,test_target.shape)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_data , train_target)
KNeighborsClassifier()

predict_target = model.predict(test_data)
print (predict_target)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_target, predict_target)
print (accuracy*100)


import joblib
joblib.dump(model, 'Oshadha_1St_AI.sav')