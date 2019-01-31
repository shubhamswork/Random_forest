# Decision Tree Regresion
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Datasets
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,[1]].values
Y=dataset.iloc[:,[2]].values

# fitting 
from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=10000)
regressor.fit(X,Y)

# predicting
y_pred=regressor.predict(6.5)

# visualising results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")

