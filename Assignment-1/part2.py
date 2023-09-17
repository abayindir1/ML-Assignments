import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.metrics as skm

# extracting data from training file
data = pd.read_excel("training.xlsx")
# x1 = data["X1 transaction date"].values
# x2 = data["X2 house age"].values
# x3 = data["X3 distance to the nearest MRT station"].values
# x4 = data["X4 number of convenience stores"].values
y = data["Y house price of unit area"].values
x=data[["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores"]]

# declaring linear regression model and training it
model = sklm.LinearRegression()
model.fit(x, y)

# extracting testing data from test data file 
test_data = pd.read_excel("test-data.xlsx")
# test_x1 = data["X1 transaction date"].values
# test_x2 = data["X2 house age"].values
# test_x3 = data["X3 distance to the nearest MRT station"].values
# test_x4 = data["X4 number of convenience stores"].values
test_y= data["Y house price of unit area"].values
test_x=data[["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores"]]

prediction = model.predict(test_x)

mse = skm.mean_squared_error(test_y, prediction)
r_square = skm.r2_score(test_y, prediction)

print(f"r squared value on the test data: {r_square}")
print(f"mean squared error on the test data: {mse}")