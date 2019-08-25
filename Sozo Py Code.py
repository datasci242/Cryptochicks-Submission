import numpy as np
import pandas as pd
import sklearn
import pickle
import matplotlib.pyplot as pyplot
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

# prompts
name = input("Hi! What's your name?")
location_area = input("Please enter your area of residence. Example: Sea Grapes, Marathon, Coral Harbour, etc...")
income = input("How much do you make per month")
age = input("How old are you?")
saving_scale = input("On a scale from 1-10, 1 being perfectly comfortable with how you spend"
                     " and 10 being near extreme couponing level of saving. ")

# Load in the Data
data = pd.read_csv("cost-of-living-2018.csv", sep=",")

data = data[["Cost of Living Index", "Rent Index", "Cost of Living Plus Rent Index", "Groceries Index",
             "Restaurant Price Index"]]

print(data.head())

predict = "Cost of Living Plus Rent Index"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0

# Run training for 200 cycles
for _ in range(200):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

if acc > best:
    best = acc
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


style.use("ggplot")
p = "Groceries Index"
# Scatter plot
pyplot.scatter(data[p], data["Cost of Living Plus Rent Index"])
pyplot.xlabel(p)
pyplot.ylabel("Average Cost of Groceries")
pyplot.show()





