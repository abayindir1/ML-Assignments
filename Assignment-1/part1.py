#Data source: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
import numpy as np
import pandas as pd

# extracting data from excel file from training excel file
data = pd.read_excel("training.xlsx")
x1 = data["X1 transaction date"].values
x2 = data["X2 house age"].values
x3 = data["X3 distance to the nearest MRT station"].values
x4 = data["X4 number of convenience stores"].values
y= data["Y house price of unit area"].values

print(x1)

# the gradient for my situation
def ssr_gradient(x, y, w):
    res = w[0] + w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] * x[3] - y
    gradient_w0 = res.mean()
    gradient_w1 = (res * x[0]).mean()
    gradient_w2 = (res * x[1]).mean()
    gradient_w3 = (res * x[2]).mean()
    gradient_w4 = (res * x[3]).mean()
    return gradient_w0, gradient_w1, gradient_w2, gradient_w3, gradient_w4

# generic method to minimize ANY convex function in the world
def gradient_descent(
     gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06
 ):
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * np.array(gradient(x, y, vector))
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector

# initial w values
initial_w = np.array(0.5, 0.5, 0.5, 0.5, 0.5)

# training the model
train = gradient_descent(ssr_gradient, np.column_stack((x1, x2, x3, x4)), y, start=initial_w)

print(train)
