

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics  # matplotlib inline

dataset = pd.read_csv("Summary of Weather.csv")
dataset.head()

# print(dataset.shape)
# print(dataset.describe())

# relationship between variables
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()


# average max temp
# plt.figure(figsize=(15, 10))
# plt.tight_layout()
# seabornInstance.displot(dataset['MaxTemp'])


# splitting the dataset
X = dataset['MinTemp'].values.reshape(-1, 1)
Y = dataset['MaxTemp'].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# import linear regression class
regressor = LinearRegression()
regressor.fit(X_train, Y_train)  # training the algorithm

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Atual': Y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(df)


# prediction graph
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# scatter diagram
plt.scatter(X_test, Y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# calculating MAE,MSE, and RMSE
print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(
    metrics.mean_squared_error(Y_test, y_pred)))
