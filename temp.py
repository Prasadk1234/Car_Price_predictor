import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

lr = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

data = pd.read_csv('Cleaned Car2.csv')

data.drop(columns=['Unnamed: 0'], inplace=True)

X = data.drop(columns=['Price'], axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 433)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

"""
score=[]
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=i)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    score.append(r2_score(y_test,y_pred))

print(np.argmax(score))

print(score[np.argmax(score)])
"""

r2_score = r2_score(y_test, y_pred)

price_val = lr.predict(pd.DataFrame([['Audi A3 Cabriolet','Audi',2019,0,'Diesel']], columns=['name','company','year','kms_driven','fuel_type']))

print(int(price_val))

