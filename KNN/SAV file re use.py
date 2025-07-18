import joblib
model = joblib.load('Oshadha_1St_AI.sav')

test=[[5.1,3.5,1.4,0.2]]
model = model.predict(test)
print(model)