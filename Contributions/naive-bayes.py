import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error   





#### Your Code Goes Here #### 

## Step 1: Load Data from CSV File ####
datafram = pd.read_csv("titanic.csv")

datafram = datafram.drop(["Name"],axis= 1)
print(datafram.describe())

## Step 2: Plot the Data ####

ages = datafram["Age"].values
fares = datafram["Fare"].values
survived = datafram["Survived"].values
colors =[]
for item in survived:
    if item == 0:
        colors.append('red')
    else:
        colors.append("blue")

# plt.scatter(ages,fares,s=50,color = colors)
# plt.show()

## Step 3: Build a NB Model ####
Features = datafram.drop(['Survived'],axis=1).values
Targets = datafram['Survived'].values
Features_Train,Targets_train = Features[0:710], Targets[0:710]
Features_test, Targets_test = Features[710:], Targets[710:]

model = GaussianNB()
model .fit(Features_Train,Targets_train)
## Step 4: Print Predicted vs Actuals ####
predictd_values = model.predict(Features_test)

for item in zip(Targets_test,predictd_values):
    print("Actual was:",item[0],"Predicted was:",item[1])
plt.plot(Features_Train,Targets_train,'o')
plt.show()
## Step 5: Estimate Error ####
print(model.score(Features_test,Targets_test))