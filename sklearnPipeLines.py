# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# /Users/juanjosebonilla/Desktop/Sistemas/BsCurso/tallerModelación

import pandas as pd
import numpy as np


#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Metrics

from sklearn.metrics import classification_report


#Selectors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



#Preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

#Pipelines
from sklearn.pipeline import Pipeline


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

imp = Imputer(missing_values="NaN",strategy="mean",axis=0)


seguro = pd.read_csv("/Users/juanjosebonilla/Desktop/Sistemas/BsCurso/tallerModelación/Seguro1.csv",delimiter=";")
validacion = pd.read_csv("/Users/juanjosebonilla/Desktop/Sistemas/BsCurso/tallerModelación/SEGURO_validacion.csv",delimiter=";")

#################### EDA
cols = seguro.columns
seguro.info()
seguro.describe()
num_cols = seguro._get_numeric_data().columns
cat_binary = [x for x in num_cols if(len(np.unique(seguro[x].values))== 2)]
num_cols = list(set(num_cols) - set(cat_binary))
seguro[num_cols] =  seguro[num_cols].fillna(seguro[num_cols].mean()).values
seguroNum = seguro[num_cols]
cat_cols = list(set(cols) - set(num_cols))


#Check for correlations

correlationsNum = get_top_abs_correlations(seguro[num_cols], n=20)

# Correlacion with dependent var

InsCor = seguro[seguro._get_numeric_data().columns].corr()
InsCor = InsCor["Ins"]
InsCor = InsCor.reset_index(name="Cor")
InsCor["Direction"] = "Positive"
InsCor["abs"] = abs(InsCor["Cor"])
InsCor.loc[InsCor["Cor"]< 0,["Direction"]] = "Negative"
InsPositive = InsCor[(InsCor["Direction"]=="Positive") & (abs(InsCor["Cor"]) >= 0.11)].sort_values(by=["abs"],ascending=False)
InsNegative = InsCor[(InsCor["Direction"]=="Negative") & (abs(InsCor["Cor"]) >= 0.11)].sort_values(by=["abs"],ascending=False)
topPositiveCols = list([x for x in list(InsPositive["index"].values) if x != "Ins"])
topNegativeCols = list(InsNegative["index"].values)


seguro = pd.get_dummies(seguro,drop_first=True)

X = seguro[["CRScore" , "MMBal", "SavBal" , "CDBal", "DDABal", "IRA", "DDA" , "Dep" ,"ATM","Branch_B14" , "Branch_B15" , "Branch_B16"]]
y = seguro["Ins"].values


validacion[num_cols] =  validacion[num_cols].fillna(seguro[num_cols].mean()).values
validacion = pd.get_dummies(validacion,drop_first=True)

X_1 = validacion.values
X_1 = validacion[["CRScore" , "MMBal", "SavBal" , "CDBal", "DDABal", "IRA", "DDA" , "Dep" ,"ATM","Branch_B14" , "Branch_B15" , "Branch_B16"]]



#####################logistic Regression


logreg = LogisticRegression()
steps = [('pca', decomposition.PCA()),("scaler",StandardScaler()),("logistic_regression",logreg)]

pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)
print(classification_report(y_test,y_pred))



#####################Decision Tree

steps = [('pca', decomposition.PCA()),("scaler",StandardScaler()),("clf",DecisionTreeClassifier())]
pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)
print(classification_report(y_test,y_pred))


#####################SVC --Modelo Elegido

steps = [("scaler",StandardScaler()),("SVM",SVC())]
pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)
print(classification_report(y_test,y_pred))




##############HyperParam Tunning## 
steps = [("scaler",StandardScaler()),("SVM",SVC())]
#steps = [('pca',decomposition.PCA()),("scaler",StandardScaler()),("SVM",SVC())]
pipeline = Pipeline(steps)
parameters= {'SVM__C':[1, 10, 100],'SVM__gamma':[0.1, 0.01]}
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(X_train,y_train)
y_pred = cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test,y_test))
print(classification_report(y_test,y_pred))




##################KNeighbors

steps = [('pca',decomposition.PCA()),("scaler",StandardScaler()),("knn",KNeighborsClassifier())]
pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)
print(classification_report(y_test,y_pred))

##############HyperParam Tunning##

steps = [('pca',decomposition.PCA()),("scaler",StandardScaler()),("knn",KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters= {"knn__n_neighbors":np.arange(1,20)}
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(X_train,y_train)
y_pred = cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test,y_test))
print(classification_report(y_test,y_pred))




