import numpy as np 
import pandas as pd 
from keras.models import load_model, Sequential
from keras.layers import Dense 



def assign_sex(row):
	if row['Sex'] == 'male':
		return 1
	else:
		return 0


test_data = pd.read_csv("test.csv", sep=",")



test_data = test_data.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket", "Fare"], axis=1)


test_data['Sex'] = test_data.apply(assign_sex, axis=1)

test_data.fillna(0, inplace=True)

X_test = np.array(test_data)




model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(5,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
			   optimizer='adam',
			   metrics=['accuracy'])


model = load_model("titanic.model")

predictions = np.round(model.predict(X_test))

predictions = np.array(predictions, dtype=np.int64)

print predictions.shape

