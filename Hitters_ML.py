
### Salary Prediction with Machine Learning

# Business Problem
# Can a machine learning project be implemented to estimate the salaries of baseball players whose salary information and career statistics for 1986 are shared?

# Dataset Story
# This dataset was originally taken from the StatLib library at Carnegie Mellon University.
# The dataset is part of the data used in the 1988 ASA Graphics Section Poster Session.
# Salary data originally from Sports Illustrated, April 20, 1987.
# 1986 and career statistics are from the 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.


# Variables
# AtBat: Number of hits with a baseball bat during the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points he earned for his team in the 1986-1987 season
# RBI: Number of players jogged,  when a batsman hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during a player's career
# CHits: The number of hits the player has taken throughout his career
# CHmRun: The player's most valuable hit during his career
# CRuns: Points earned by the player during his career
# CRBI: The number of players the player has made during his career
# CWalks: Number of mistakes made by the opposing player during the player's career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with levels E and W indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate in-game
# Assists: Number of assists made by the player in the 1986-1987 season
# Errors: Player's number of errors in the 1986-1987 season
# Salary: The salary of the player in the 1986-1987 season (over thousand)
# NewLeague: a factor with A and N levels indicating the player's league at the start of the 1987 season


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


############################################
# Exploratory Data Analysis(EDA)
############################################

# Lets take a general look to data
df = pd.read_csv(r'C:/Users/Aybüke Hamide AK/Desktop/DSMLBC6/VBO-WEEK 8/Ders Notları/hitters.csv')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# As you can see in the results, we have 56 NA values in our target variable

# DEPENDENT VARIABLE ANALYSIS
# We checked our dependent variable "Salary" with graphs to see the distribution of data.

df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()

sns.boxplot(df["Salary"])
plt.show()

# DETERMINING CATEGORICAL AND NUMERICAL VARIABLES

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# We have 3 categorical and 17 numeric columns

# Observations: 322
# Variables: 20
# cat_cols: 3
# num_cols: 17
# cat_but_car: 0
# num_but_cat: 0

# CATEGORICAL VARIABLE ANALYSIS

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Salary", cat_cols)

# We examine the numerical distribution of categorical variable classes and their ratios relative to each other.

# League : 2
#    COUNT  RATIO  TARGET_MEAN
# A    175  0.543      542.000
# N    147  0.457      529.118
# Division : 2
#    COUNT  RATIO  TARGET_MEAN
# E    157  0.488      624.271
# W    165  0.512      450.877
# NewLeague : 2
#    COUNT  RATIO  TARGET_MEAN
# A    176  0.547      537.113
# N    146  0.453      534.554


# NUMERICAL VARIABLE ANALYSIS

# Here, by examining the distribution of the numeric variable in the data, its maximum and minimum values,
# we get information before performing outlier analysis.

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=False)

# We should remove our target column from numerical columns list.
num_cols.remove("Salary")

# OUTLIERS ANALYSIS

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.1, q3=0.9))

# AtBat False
# Hits False
# HmRun False
# Runs False
# RBI False
# Walks False
# Years False
# CAtBat False
# CHits True
# CHmRun True
# CRuns False
# CRBI False
# CWalks True
# PutOuts False
# Assists False
# Errors False
# Salary False

df.describe()

# We have some outliers in our "CHits",  "CHmRun", "CWalks" columns. And we want to change outliers to thresholds.
# We can not directly remove them, due to we don't want to decrease our data

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outlier(df, col, q1=0.1, q3=0.9):
        replace_with_thresholds(df, col, q1=0.1, q3=0.9)



# NA observation ANALYSIS

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


df.shape # (322, 20)

missing_values_table(df, True)

#         n_miss  ratio
# Salary      59 18.320

# We have 59 NA values in our target column, now we should remove them

df.dropna(inplace=True)

df.shape # (263, 20)

## CORRELATION MATRIX
def correlated_map(dataframe, plot=False):
    corr = dataframe.corr()
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="YlGnBu", annot=True, linewidths=.7)
        plt.xticks(rotation=60, size=15)
        plt.yticks(size=15)
        plt.title('Correlation Map', size=20)
        plt.show()

correlated_map(df, plot=True)

def target_correlation_matrix(dataframe, corr_th=0.5, target="Salary"):
    """
    Returns the variables that have a correlation above the threshold value given with the dependent variable.
    :param dataframe: dataframe
    :param corr_th: threshold
    :param target:  name of dependent variable
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("higher threshold, decrease your threshold, value of corr_th!")


target_correlation_matrix(df, corr_th=0.5, target="Salary")

# We can see in the matrix,the correlation between CAtBat and CHits is 1.
# Therefore, we may remove one of them or we can analysis
# df.drop("CAtBat", axis=1)
# df.drop("CHits", axis=1)

# FEATURE EXTRACTION

## Player success hit rate
df['NEW_Success_CHit'] = df['CHits'] / df['CAtBat']  # total career success hits / total career hits
## Success hit rate in the 1986-1987 season
df['NEW_Success_86_87_Hits'] = df['Hits'] / df['AtBat'] # total success hits in 1986-1987 / total hits in 1986-1987
## The point player earned per hit in 1986-1987
df['NEW_Success_86_87_Hits'] = df['Runs'] / df['AtBat'] # The points player earned for his team in the 1986-1987 season / number of hits in the 1986-1987 season
# Number of misses in the 1986-1987 season
df["Hits_Success"] = (df["AtBat"] - df["Hits"])
## Average number of points,the player earned per year
df["NEW_CRUNS_RATE"] = df["CRuns"] / df["Years"]
## Average number of success hits per year
df["NEW_CHITS_RATE"] = df["CHits"] / df["Years"]
## The ratio of the number of players when a batsman hit, to player's career time
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years'] # Number of players run when a batsman hit / career year

## We divided it into 3 categorical classes according to helping your teammate during the game.
Putouts_label = ["little_helper", "medium_helper", "very_helper"]
df["NEW_PUTOUTS_CAT"] = pd.qcut(df["PutOuts"], 3, labels=Putouts_label)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum().sum()


# Observations: 263
# Variables: 27
# cat_cols: 4
# num_cols: 23
# cat_but_car: 0
# num_but_cat: 1

# ONE-HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head(10)


from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score



#final_df = df.copy()

final_df = df

############################################
# MODELING
############################################
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate


y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# RMSE: 342.4832 (LR)
# RMSE: 343.6933 (Ridge)
# RMSE: 341.8502 (Lasso)
# RMSE: 338.4269 (ElasticNet)
# RMSE: 328.6061 (KNN)
# RMSE: 410.4488 (CART)
# RMSE: 267.7454 (RF)
# RMSE: 444.3155 (SVR)
# RMSE: 268.7408 (GBM)
# RMSE: 283.0765 (XGBoost)
# RMSE: 288.7831 (LightGBM)


############################################
# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 110, num = 40)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 6, 8, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

#{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt'], 'max_depth': [2, 4, 7, 10, 13, 15, 18, 21, 24, 26, 29, 32, 35, 38, 40, 43, 46, 49, 51, 54, 57, 60, 62, 65, 68, 71, 74, 76, 79, 82, 85, 87, 90, 93, 96, 98, 101, 104, 107, 110, None], 'min_samples_split': [2, 3, 5, 6, 8, 10, 15, 20], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
best_param = rf_random.best_params_

# {'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 32, 'bootstrap': True}

# Train RMSE


y = final_df["Salary"]
X = final_df.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=16)


rf = RandomForestRegressor(n_estimators = 200, min_samples_split= 3, min_samples_leaf= 1, max_features= "sqrt", max_depth= 32)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

## 110.97663363570734

df["Salary"].mean()
df["Salary"].std()

# Test RMSE
y_pred = rf.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

## 224.1628904504917



######################################################
# Automated Hyperparameter Optimization
######################################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2,30)}

rf_params = {"max_depth": [3, 5, 8, 10, 15, 18, 20, 30, 32, None],
             "max_features": [3, 5, 7, "auto", "sqrt"],
             "min_samples_split": [2, 3, 5, 6, 8, 10],
             "n_estimators": [50, 100, 200, 300, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [3, 5, 6, 7,  9, 12],
                  "n_estimators": [40, 50, 75, 100, 150, 200, 300],
                  "colsample_bytree": [0.3, 0.5, 0.6, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.1, 0.01, 0.001],
                   "n_estimators": [300, 500, 700, 750, 800, 850, 900, 950, 1000, 1500],
                   "colsample_bytree": [0.2, 0.3, 0.5, 0.6, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


# ########## CART ##########
# RMSE: 377.0022 (CART)
# RMSE (After): 318.2927 (CART)
# CART best params: {'max_depth': 3, 'min_samples_split': 14}
# ########## RF ##########
# RMSE: 257.6705 (RF)
# RMSE (After): 251.6585 (RF)
# RF best params: {'max_depth': 32, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 200}
# ########## XGBoost ##########
# RMSE: 282.9783 (XGBoost)
# RMSE (After): 253.5331 (XGBoost)
# XGBoost best params: {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# ########## LightGBM ##########
# RMSE: 277.1542 (LightGBM)
# RMSE (After): 264.8924 (LightGBM)
# LightGBM best params: {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'n_estimators': 1000}

