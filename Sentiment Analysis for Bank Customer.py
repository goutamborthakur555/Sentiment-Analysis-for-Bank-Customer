import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

os.getcwd()
os.chdir("...location\\FileName")

train = pd.read_csv("....\\train.csv")
test = pd.read_csv("....\\test.csv")

train.shape # (76020, 371)
test.shape # (75818, 370)

# Creating a "TARGET" column in test dataset and assigning NA values to it to match the no. of columns with train dataset.
test["TARGET"] = np.nan
test.columns
test.shape
#(75818, 371)

train.columns

# Increase the print output
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# Checking for Missing Value
train.isnull().sum() # No NA's
test.isnull().sum() # No NA's excep the "TARGET" variable
# Hence, missing value imputation is not required.

train.dtypes # float64 & int64
test.dtypes # float64 & int64
# Hence, dummy variable creation is not required.

Train_Summary = train.describe()
# No Scaling is required as units are not so high

# %split of "TARGET"
split_per_target = train["TARGET"].value_counts()/len(train["TARGET"]) * 100
# =============================================================================
# 0    96.043147
# 1     3.956853
# #######################
# 0    73012
# 1     3008
# =============================================================================

#Visualization of "TARGET" distribution in terms of historgram
train.TARGET.hist(bins=2)
plt.title("TARGET Column Distribution")
plt.xlabel("TARGET")
plt.ylabel("Records")

# Combine train and test
FullRaw = pd.concat([train, test], axis = 0)

#Divide train further into X-Indepenedent and Y-Dependent Variables and sampling using Stratified Sampling
X = train.drop(['TARGET'], axis = 1).copy()
y = train['TARGET'].copy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123, stratify = y, test_size = 0.30) #70-30%

# =============================================================================
# # Pipelines Creation
#  1. Data Preprocessing by using Standard Scaler
#  2. Reduce Dimension using PCA
#  3. Apply  Classifier
# =============================================================================

pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                      ('pca1',PCA(n_components=185)),
                      ('lr_classifier',LogisticRegression(random_state=0))])

pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                      ('pca2',PCA(n_components=185)),
                      ('dt_classifier',DecisionTreeClassifier())])

pipeline_rf=Pipeline([('scalar3',StandardScaler()),
                                ('pca3',PCA(n_components=185)),
                                ('rf_classifier',RandomForestClassifier())])

# Making the list of pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf]

best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))

# =============================================================================
# Logistic Regression Test Accuracy: 0.95970358677541
# Decision Tree Test Accuracy: 0.9208103130755064
# RandomForest Test Accuracy: 0.9568534596158905
# =============================================================================

# =============================================================================
# for i,model in enumerate(pipelines):
#     if model.score(X_test,y_test)>best_accuracy:
#         best_accuracy=model.score(X_test,y_test)
#         best_pipeline=model
#         best_classifier=i
# print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))
# =============================================================================

# Pipelines Perform Hyperparameter Tuning Using Grid SearchCV
from sklearn.model_selection import GridSearchCV

# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
        {"classifier": [LogisticRegression()],
                        "classifier__penalty": ['l2','l1'],
                        "classifier__C": np.logspace(0, 4, 10)
                        },
        {"classifier": [LogisticRegression()],
                        "classifier__penalty": ['l2'],
                        "classifier__C": np.logspace(0, 4, 10),
                        "classifier__solver":['newton-cg','saga','sag','liblinear']
                        },
        {"classifier": [RandomForestClassifier()],
                        "classifier__n_estimators": [10, 100, 1000],
                        "classifier__max_depth":[5,8,15,25,30,None],
                        "classifier__min_samples_leaf":[1,2,5,10,15,100],
                        "classifier__max_leaf_nodes": [2, 5,10]}]

# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=2,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train,y_train)

print(best_model.best_estimator_)
# =============================================================================
# Pipeline(memory=None,
#          steps=[('classifier',
#                  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
#                                         class_weight=None, criterion='gini',
#                                         max_depth=25, max_features='auto',
#                                         max_leaf_nodes=10, max_samples=None,
#                                         min_impurity_decrease=0.0,
#                                         min_impurity_split=None,
#                                         min_samples_leaf=2, min_samples_split=2,
#                                         min_weight_fraction_leaf=0.0,
#                                         n_estimators=10, n_jobs=None,
#                                         oob_score=False, random_state=None,
#                                         verbose=0, warm_start=False))],
#          verbose=False)
# =============================================================================

print("The mean accuracy of the model is:",best_model.score(X_test,y_test))
#The mean accuracy of the model is: 0.9604490046478997

# MakePipelines In SKLearn
from sklearn.pipeline import make_pipeline

# Create a pipeline
pipe = make_pipeline((RandomForestClassifier()))
# Create dictionary with algorithms and their hyperparameters
grid_param = [{"classifier": [RandomForestClassifier()],
                        "classifier__n_estimators": [10],
                        "classifier__max_depth":[25],
                        "classifier__min_samples_leaf":[2],
                        "classifier__max_leaf_nodes": [10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=2, n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train,y_train)

best_model.score(X_test,y_test)
#0.9604490046478997 ; RandomForest




