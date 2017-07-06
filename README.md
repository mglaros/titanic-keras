# titanic-keras

Created a Neural Network with Keras for the Titanic Survival Prediction Dataset.

The dataset was given in the Kaggle competition:

https://kaggle.com/c/titanic

where, given some passenger data the algorithm will predict who will die and who will survive by using some ML algorithms.

In this case, a dense (fully-connected) Neural Network with two hidden layers was used. Survival is treated as a binary variable: 0 if dead and 1 if survived.

Input:

Train.csv

PassengerId - a number giving the unique ID of the passenger
Survived - survival (0 = No, 1 = Yes)
Pclass - passenger class (1 to 3)
Name - passenger full name
Sex - male or female
Age - age of the passenger
SibSp - number of siblings or spouses on board 
Parch - number of parents or children on board
Ticket - ticket number
Fare - ticket fare
Cabin - (if applicable) cabin in the ship
Embarked - port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Test.csv

Same as Train.csv, but with Survival column omitted



The features that were used to do the training were Pclass, Sex, Age, SibSp, and Parch


To run the program type:  **python titanic_train.py** followed by **python predict_survival.py** which will read in the model file **titanic.model** produced by the training script to make predictions on the Test.csv file returning an array of 0s and 1s corresponding to the predicted death or survival of the passenger under consideration.

