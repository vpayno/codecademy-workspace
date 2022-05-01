import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
print(f"prod per year:\n{prod_per_year}")
print()

X = df["year"]
X = X.values.reshape(-1,1)
print("     year:")
print(f"{X}")
print()

Y = df["totalprod"]
print(f"totalprod:")
print(f"{Y}")
print()

plt.scatter(X, Y)
plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print(f"    slope: {regr.coef_[0]}")
print(f"intercept: {regr.intercept_}")
print()

Y_predict = regr.predict(X)
print(f"predict(X): {Y_predict}")
print()

plt.plot(X, Y_predict)
plt.show()

X_future = np.array(range(2013, 2050))
print(f"X_future: {X_future}")
X_future = X_future.reshape(-1, 1)
print(f"X_future: {X_future}")
print()

future_predict = regr.predict(X_future)
print(f"predict(X_future): {future_predict}")
print()

plt.plot(X_future, future_predict)
plt.show()
