# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import necessary libraries such as `numpy`, `pandas`, and `StandardScaler` from `sklearn.preprocessing` for scaling the data.

2. **Define Linear Regression Function**: Implement the `linear_regression()` function:
   - Add a column of ones to the input data (`X1`) to account for the intercept term.
   - Initialize the parameter vector `theta` with zeros.
   - Perform gradient descent:
     - Compute the predictions by multiplying the input features with `theta`.
     - Calculate the error as the difference between predictions and actual values.
     - Update `theta` using the gradient descent formula.

3. **Load Dataset**: Load the `50_Startups.csv` dataset using `pandas`. Extract the input features (`X`) and output labels (`y`).

4. **Convert Data to Floats**: Convert the input features `X` and output `y` to floating-point numbers for scaling and computations.

5. **Scale the Features and Target**: Use `StandardScaler` to scale the input features (`X1_Scaled`) and the output target (`Y1_Scaled`) to normalize the data for gradient descent optimization.

6. **Train the Linear Regression Model**: Call the `linear_regression()` function with scaled data (`X1_Scaled` and `Y1_Scaled`), and return the optimized parameter vector `theta`.

7. **Prepare New Data for Prediction**: Define a new data point (`new_data`), reshape it, and scale it using the `StandardScaler`.

8. **Make Prediction**: Use the learned `theta` to predict the output for the new scaled input data by calculating the dot product of the input with `theta`. Then, inverse-transform the prediction to get the original scale value.

9. **Display the Prediction**: Print the predicted value after inverse transforming the result back to its original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pandidharan.G.R
RegisterNumber:  212222040111
*/
```

```PYTHON
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01,num_iters=1000):

  X = np.c_[np.ones(len(X1)), X1] #add a column of ones to X for the intercept term
  theta = np.zeros(X.shape[1]).reshape(-1,1) #initialize theta with zeros

#perform gradient descent
  for _ in range(num_iters):

    predictions = (X).dot(theta).reshape(-1,1)
    errrors = (predictions - y).reshape(-1,1)

    theta = learning_rate * (1/len(X)) * (X.T).dot(errrors)

  return theta

data = pd.read_csv('/content/50_Startups.csv')
X = data.iloc[1:,:-2].values
print(X)
X1 = X.astype(float)
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
scaler = StandardScaler()
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2024-09-05 212801](https://github.com/user-attachments/assets/4474501a-ca95-4a0e-933a-9692af1f7c7c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
