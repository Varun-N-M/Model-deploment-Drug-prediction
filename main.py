import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import pickle

df = pd.read_csv('drug200.csv')
print(df.head().to_string())

print(df.info())
print(df['Drug'].value_counts())
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['BP'] = df['BP'].map({'LOW': 1, 'NORMAL': 2, 'HIGH': 3})
df['Cholesterol'] = df['Cholesterol'].map({'NORMAL':1,'HIGH':2})
df['Drug'] = df['Drug'].map({'drugA':1,'drugB':2,'drugC':3,'drugX':4,'DrugY':5})
# df['Drug'] = le.fit_transform(df['Drug'])
print(df.head().to_string())
print(df['Drug'].value_counts())

x = df.drop('Drug',axis=1)
x = x.values
y = df['Drug']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

pickle.dump(dt,open('model.pkl','wb'))
# model = pickle.load(open('model.pkl','rb'))
#
# old_y = df['Drug']
# print('actual:',le.inverse_transform(old_y)[1])
#
# new_inp = np.array(df.drop('Drug',axis=1))
# new_y = model.predict(new_inp[1].reshape(1,-1))
#
#
# print('Prediction',le.inverse_transform(new_y))