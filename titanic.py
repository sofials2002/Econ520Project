import numpy as np
import yfinance as yf
import csv
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


raw = []
with open('titanic.csv', newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        raw += [row]

data = np.array(raw[1:])

names = data[:, 0]

male = data[:, 1] == "Male"
age = data[:, 2].astype(int)
young_adult = data[:, 3].astype(int)
middle_age = data[:, 4].astype(int)
old = data[:, 5].astype(int)
first_class = data[:, 7].astype(int)
survive = data[:, 8].astype(int)


inputs = sm.add_constant(np.array([male, young_adult, middle_age, old, first_class]).T)
model = sm.OLS(survive, inputs).fit()
print(model.summary())
print("male, young_adult, middle_age, old, first_class")

plt.hist(age, bins=25)
plt.show()
