import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

profile_data=pd.read_csv("/content/drive/MyDrive/SEM 6/tcss555/training/profile/profile.csv",index_col=0)

profile_data.rename(columns={"userid":"userId"},inplace=True)
liwc_features=pd.read_csv("/content/drive/MyDrive/SEM 6/tcss555/training/LIWC/LIWC.csv")
liwc_features

temp=liwc_features.drop("Seg",axis=1)
temp.columns

# Split the features into two sets: set1 and set2
set1 = temp.iloc[:, :81]
set2 = profile_data.iloc[:,1:8]

X = set1  # Adjust column names accordingly
y = set2["ope"]

scaler = StandardScaler()
# Fit the scaler on your data and transform it
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Apply PCA for dimensionality reduction
n_components = 63  # Adjust the number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Regression models
linear_reg_model = LinearRegression()
# Create and train the Elastic Net model
alpha = 0.5  # You can adjust this parameter based on your needs
l1_ratio = 0.5  # You can adjust this parameter based on your needs
elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
random_forest_model = RandomForestRegressor()
svr = SVR(kernel='linear', C=1.0)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)

# Fit the models
linear_reg_model.fit(X_train_pca, y_train)
random_forest_model.fit(X_train_pca, y_train)
elastic_net_model.fit(X_train_pca,y_train)
svr.fit(X_train_pca, y_train)
ridge.fit(X_train_pca,y_train)
lasso.fit(X_train_pca,y_train)

# Predictions
y_pred_linear_reg = linear_reg_model.predict(X_test_pca)
y_pred_random_forest = random_forest_model.predict(X_test_pca)
y_pred_elastic_net_model=elastic_net_model.predict(X_test_pca)
y_pred_svr = svr.predict(X_test_pca)
y_pred_ridge=ridge.predict(X_test_pca)
y_pred_lasso=lasso.predict(X_test_pca)
# Evaluate performance
mse_linear_reg = mean_squared_error(y_test, y_pred_linear_reg)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mse_elastic_net=mean_squared_error(y_test, y_pred_elastic_net_model)
mse_svr=mean_squared_error(y_test, y_pred_svr)
mse_ridge=mean_squared_error(y_test,y_pred_ridge)
mse_lasso=mean_squared_error(y_test,y_pred_lasso)

# Print MSE values
#print(f"Root Mean Squared Error (Linear Regression): {mse_linear_reg**0.5}")
#print(f"Root Mean Squared Error (Random Forest): {mse_random_forest**0.5}")
print(f"Root Mean Squared Error (Elasticnet): {mse_elastic_net**0.5}")
print(f"Root Mean Squared Error (Ridge Regressor): {mse_ridge**0.5}")
print(f"Root Mean Squared Error (Lasso Regressor): {mse_lasso**0.5}")
print(f"Root Mean Squared Error (SV Regressor): {mse_svr**0.5}")


'''
# Visualize predicted vs actual personality scores
plt.scatter(y_test, y_pred_linear_reg, label='Linear Regression')
plt.scatter(y_test, y_pred_random_forest, label='Random Forest')
plt.xlabel('Actual Personality Scores')
plt.ylabel('Predicted Personality Scores')
plt.legend()
plt.show()
'''

#XGB WITH GRIDSEARCH CV
from sklearn.model_selection import GridSearchCV


# Define the XGBoost model
model = XGBRegressor()

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],  # Add gamma for regularization
    'min_child_weight': [1, 3, 5],  # Add min_child_weight for regularization
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_result = grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_result.best_params_

# Use the best parameters to create the final model
final_model = XGBRegressor(**best_params)

# Train the final model on the entire training set
final_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = final_model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Optionally, you can also access other information from grid_result, like grid_result.cv_results_, grid_result.best_score_, etc.


#Treying out neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


X = set1  # Adjust column names accordingly
y = set2["ope"]

scaler = StandardScaler()
# Fit the scaler on your data and transform it
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Apply PCA for dimensionality reduction
n_components = 63  # Adjust the number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Neural Network Model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_pca.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

X = set1  # Adjust column names accordingly
y = set2["con"]

scaler = StandardScaler()
# Fit the scaler on your data and transform it
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Apply PCA for dimensionality reduction
n_components = 63  # Adjust the number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#RandomForest + XGB:
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# Define base models
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42))
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)
print(f'Neural Network Root Mean Squared Error: {mse**0.5}')


# Linear Regression + ElasticNet + Ridge + Lasso

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso

# Define base models
base_models = [
    ('linear', LinearRegression()),
    ('elastic_net', ElasticNet(alpha=0.1)),
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1))
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)
print(f'Neural Network Root Mean Squared Error: {mse**0.5}')

#Random Forest + Support Vector Regressor (SVR)


from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Define base models
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR())
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)
print(f'Neural Network Root Mean Squared Error: {mse**0.5}')

#XGBoost + Linear Regression + RandomForest

from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Define base models
base_models = [
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42)),
    ('linear', LinearRegression()),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42))
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)
print(f'Neural Network Root Mean Squared Error: {mse**0.5}')


#XGB+LINREG+LASSO+RIDGE+RFR+SVR

from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Define base models
base_models = [
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42)),
    ('linear', LinearRegression()),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('elastic_net', ElasticNet(alpha=0.1)),
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1)),
    ('svr', SVR()),
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)
print(f'Neural Network Root Mean Squared Error: {mse**0.5}')
