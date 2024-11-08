# %% [markdown]
# # 1. Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier

# Model Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Model Saving
import pickle

# %% [markdown]
# # 2. Data Loading

# %%
data_agriculture = pd.read_csv("data/Agriculture_Dataset/agriculture_dataset.csv")
display(data_agriculture)
display(data_agriculture.info())

# %%
data_soil = pd.read_csv("data/Soil_Measures/soil_measures.csv")
display(data_soil)
display(data_soil.info())

# %%
data_main = pd.read_csv("data/Crop_Yield/crop_yield.csv")
display(data_main)
display(data_main.info())

# %% [markdown]
# # 3. Data Preprocessing

# %% [markdown]
# ### 3.1 Initial Preprocessing

# %%
def one_hot_encode(df, column_name):
    try:
        df = pd.get_dummies(df, columns=[column_name], prefix=column_name, drop_first=False)
        df = df.replace({True:1,False:0})
    finally:
        return df

# %%
# Drop unused columns
try:
    data_agriculture_clean = data_agriculture.drop(["Farm_ID", "Season"], axis=1)
except:
    pass

data_agriculture_clean = data_agriculture_clean.groupby(["Crop_Type","Irrigation_Type","Soil_Type"]).mean().reset_index()

# One-hot encode Soil_Type in data_agriculture
data_agriculture_clean = one_hot_encode(data_agriculture_clean, "Soil_Type")

# Modify the irrigation data to match the main dataframe
try:
    data_agriculture_clean["Irrigation_Used"] = data_agriculture_clean["Irrigation_Type"].apply(lambda x: 0 if x == "Rain-fed" else 1)
    data_agriculture_clean = data_agriculture_clean.drop("Irrigation_Type", axis=1)
except:
    pass

data_agriculture_clean.to_csv("data_clean/agriculture_clean.csv")

# %%
data_soil_clean = data_soil.groupby("crop").mean().reset_index()
data_soil_clean.to_csv("data_clean/soil_clean.csv")

# %%
# Drop unused columns
try:
    data_main_clean = data_main.drop(["Region", "Weather_Condition"], axis=1)
except:
    pass

data_main_clean = one_hot_encode(data_main_clean, "Soil_Type")
data_main_clean = one_hot_encode(data_main_clean, "Crop")

data_main_clean = data_main_clean.replace({True:1,False:0})

data_main_clean.to_csv("data_clean/main_clean.csv")

# %% [markdown]
# ### 3.2 Correlation Matrix

# %%
corr = data_main_clean.corr()
display(corr)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, vmax=1, vmin=-1, square=True, linewidths=.5, cmap="coolwarm", cbar_kws={"shrink": .5}) # Most of these arguments are not really useful in this case
plt.show()

# %% [markdown]
# ### 3.3 Feature Scaling

# %%
X = data_main_clean.drop("Yield_tons_per_hectare",axis=1)
y = data_main_clean["Yield_tons_per_hectare"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# %%
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# %%
pickle.dump(scaler, open("scaler.pkl", "wb"))

# %% [markdown]
# # 4. Machine Learning

# %%
# Define the models and the sample size
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree_4": DecisionTreeRegressor(max_depth=4),
    "Decision Tree_5": DecisionTreeRegressor(max_depth=5),
    "Random Forest_50": RandomForestRegressor(n_estimators=50),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Random Forest_200": RandomForestRegressor(n_estimators=200),
    "KNN_3": KNeighborsRegressor(n_neighbors=3),
    "KNN_5": KNeighborsRegressor(n_neighbors=5),
    "KNN_7": KNeighborsRegressor(n_neighbors=7),
    "SVR_rbf": SVR(),
    "SVR_linear": SVR(kernel="linear"),
    "GBR": GradientBoostingRegressor(n_estimators=2000, learning_rate=0.3)
}

sample_size = 200000

# %%
# Check the R2-score of each model and choose the best-performing ones
for name, model in models.items():
    scores = cross_val_score(model, X_train.head(sample_size), y_train.head(sample_size), scoring="r2")
    print(f"{name}: R2-score:\t{np.mean(scores):.5f}")

# %%
# Linear Regression
models["Linear Regression"].fit(X_train.head(sample_size),y_train.head(sample_size))
y_pred_LN = models["Linear Regression"].predict(X_train.tail(sample_size))
r2_LN = r2_score(y_train.tail(sample_size), y_pred_LN)
print(f"R2-score for Linear Regression ({sample_size} samples): {r2_LN:.5f}")

# %%
# Random Forest 
tuned_RF = GridSearchCV(estimator=models["Random Forest"],
                        param_grid={'n_estimators': [50, 100, 150, 200],'max_depth': [10, 20, 30, None]},
                        cv=5,
                        scoring='r2'
                        )
tuned_RF.fit(X_train.head(sample_size),y_train.head(sample_size))
y_pred_RF = tuned_RF.predict(X_train.tail(sample_size))
r2_RF = r2_score(y_train.tail(sample_size), y_pred_RF)
print(f"R2-score for Random Forest ({sample_size} samples): {r2_RF:.5f}\nBest parameters: {tuned_RF.best_params_}")

# RandomForestRegressor(n_estimators=150, max_depth=10) # Results for GridSearchCV (05.11 16:25 R2:0.9098 (20k samples))

# %%
# Support Vector
tuned_SVR = GridSearchCV(estimator=models["SVR_rbf"],
                        param_grid={'kernel': ["linear", "poly", "rbf"]},
                        cv=5,
                        scoring='r2'
                        )
tuned_SVR.fit(X_train.head(sample_size),y_train.head(sample_size))
y_pred_SVR = tuned_SVR.predict(X_train.tail(sample_size))
r2_SVR = r2_score(y_train.tail(sample_size), y_pred_SVR)
print(f"R2-score for Support Vector ({sample_size} samples): {r2_SVR:.5f}\nBest parameters: {tuned_SVR.best_params_}")


# %%
tuned_SVR = SVR(kernel="linear", verbose=1) # Results for GridSearchCV (05.11 16:42 R2:0.9132 (20k samples))

# %%
# We now test the best-scoring model with the whole dataset
trained_model = tuned_SVR.fit(X_train.head(sample_size),y_train.head(sample_size))

# %%
pickle.dump(trained_model, open("trained_model.pkl", "wb"))

# %%
y_pred = trained_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2-score for Support Vector: {r2:.5f}")

