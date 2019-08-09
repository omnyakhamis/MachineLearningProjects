# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:06:14 2019

@author: omnya khamis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import statsmodels.api as sm
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler#
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV ,cross_val_predict,KFold
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

#test=pd.DataFrame({})
#--------------------------------------- Function calculateNull(dataset) --------------------------------------#
def calculateNull(dataset):
    
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    f, ax = plt.subplots(figsize=(80, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('test missing data by feature', fontsize=15)
    return missing_data

#---------------------------------  processData(file)   ---------------------------------------------------------------------------
def processData(dataset):
    
    
    dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)
    dataset['OverallQual'] = dataset['OverallQual'].apply(str)
    dataset['OverallCond'] = dataset['OverallCond'].apply(str)
   #-------------------------------  LotFrontage  -------------------------------------------------# 
    dataset["LotFrontage"] = dataset.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
   #--------------------------------Categoracal features  ---------------------------------------------------------------------------#
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])
    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])
    
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        dataset[col] = dataset[col].fillna(0)
        
    dataset = dataset.drop(['Utilities'], axis=1)
    
    dataset["Functional"] = dataset["Functional"].fillna("Typ")
    
    
    
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])
    
    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])
    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])
    dataset['Heating'] = dataset['Heating'].fillna(dataset['Heating'].mode()[0])
    dataset['RoofMatl'] = dataset['RoofMatl'].fillna(dataset['RoofMatl'].mode()[0])
    
    dataset['MSSubClass'] = dataset['MSSubClass'].fillna("None")
    dataset['OverallQual'] = dataset['OverallQual'].fillna("None")
    dataset['OverallCond'] = dataset['OverallCond'].fillna("None")

    
    
    #-------------------------------------------------------------#
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        dataset[col] = dataset[col].fillna(0)
    
     #----------  change type GarageYrBlt to int64   ------------#
    
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].astype(np.int64) 
    #-----------------  MasVnrArea  ------------------------------------------------------------#
    
    dataset["MasVnrArea"] = dataset["MasVnrArea"].fillna(0)
    
    
    
#---------------------------------- scaler --------------------------------------------------------#    
    df_scaled=dataset.select_dtypes(include=[np.number])
    #df_scaled.drop(['SalePrice'],axis=1,inplace=True)
    scaler = StandardScaler()
    
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled),columns=df_scaled.columns)
    
    #oldCatNum=dataset[['SalePrice']]
    dataset.drop(dataset.select_dtypes(include=[np.number]).columns,axis=1,inplace=True)
    
    dataset=pd.concat([df_scaled,dataset],axis=1)  
    

   
    
    
    # --------------------------   replace NA values with 'None'----------#
    
    str_cols = dataset.select_dtypes(include=['object']).columns
    dataset.loc[:, str_cols] = dataset.loc[:, str_cols].fillna('None')
    
    #--------------------------------------------------------------------------
    
    
    
    
    
    #----------------------------   Maping certains features   --------------#
    
    dataset['Street'] = dataset.Street.map({'Grvl':0,'Pave':1})
    
    dataset['Alley'] = dataset.Alley.map({'None':0,'Pave':1, 'Grvl':2})
    
    dataset['LotShape'] = dataset.LotShape.map({'IR3':0,'IR2':1, 'IR1':2 , 'Reg':3})
    
    dataset['LandContour'] = dataset.LandContour.map({'Low':0,'HLS':1, 'Bnk':2 , 'Lvl':3})
    
    #dataset['Utilities'] = dataset.Utilities.map({'ELO':0,'NoSeWa':1, 'NoSewr':2 , 'AllPub':3})
    
    dataset['LotConfig'] = dataset.LotConfig.map({'Inside':0,'Corner':1, 'CulDSac':2 , 'FR2':3 , 'FR3':4})
    
    dataset['LandSlope'] = dataset.LandSlope.map({'Sev':0,'Mod':1, 'Gtl':2})
    
    dataset['ExterQual'] = dataset.ExterQual.map({'Po':0,'Fa':1, 'TA':2 , 'Gd':3 , 'Ex':4})
    
    dataset['ExterCond'] = dataset.ExterCond.map({'Po':0,'Fa':1, 'TA':2 , 'Gd':3 , 'Ex':4})
    
    dataset['BsmtQual'] = dataset.BsmtQual.map({'None':0,'Po':1, 'Fa':2 , 'TA':3 , 'Gd':4, 'Ex':5})
    
    dataset['BsmtCond'] = dataset.BsmtCond.map({'None':0,'Po':1, 'Fa':2 , 'TA':3 , 'Gd':4, 'Ex':5})
    
    dataset['BsmtExposure'] = dataset.BsmtExposure.map({'None':0,'No':1, 'Mn':2 , 'Av':3 , 'Gd':4})
    
    dataset['BsmtFinType1'] = dataset.BsmtFinType1.map({'None':0,'Unf':1, 'LwQ':2 , 'Rec':3 , 'BLQ':4, 'ALQ':5, 'GLQ':6})
    
    dataset['BsmtFinType2'] = dataset.BsmtFinType2.map({'None':0,'Unf':1, 'LwQ':2 , 'Rec':3 , 'BLQ':4, 'ALQ':5, 'GLQ':6})
    
    dataset['HeatingQC'] = dataset.HeatingQC.map({'Po':0,'Fa':1, 'TA':2 , 'Gd':3 , 'Ex':4})
    
    dataset['CentralAir'] = dataset.CentralAir.map({'N':0,'Y':1})
    
    dataset['KitchenQual'] = dataset.KitchenQual.map({'Po':0,'Fa':1, 'TA':2 , 'Gd':3 , 'Ex':4})
    
    dataset['FireplaceQu'] = dataset.FireplaceQu.map({'None':0,'Po':1, 'Fa':2 , 'TA':3 , 'Gd':4, 'Ex':5})
    
    dataset['GarageType'] = dataset.GarageType.map({'None':0,'Detchd':1, 'CarPort':2 , 'BuiltIn':3 , 'Basment':4, 'Attchd':5, '2Types':6})
    
    dataset['GarageFinish'] = dataset.GarageFinish.map({'None':0,'Unf':1, 'RFn':2 , 'Fin':3})
    
    dataset['GarageQual'] = dataset.GarageQual.map({'None':0,'Po':1, 'Fa':2 , 'TA':3 , 'Gd':4, 'Ex':5})
    
    dataset['GarageCond'] = dataset.GarageCond.map({'None':0,'Po':1, 'Fa':2 , 'TA':3 , 'Gd':4, 'Ex':5})
    
    dataset['PavedDrive'] = dataset.PavedDrive.map({'N':0,'P':1, 'Y':2})
    
    dataset['PoolQC'] = dataset.PoolQC.map({'None':0,'Fa':1, 'TA':2 , 'Gd':3 , 'Ex':4})
    
    dataset['Fence'] = dataset.Fence.map({'None':0,'MnWw':1, 'GdWo':2 , 'MnPrv':3 , 'GdPrv':4})
    
    
    
    #-----------------------  change 'object'  column type to 'category' type --------#
    dataset[dataset.select_dtypes(['object']).columns]=dataset.select_dtypes(include=['object']).apply(lambda x: x.astype('category'))
    
    
    
    #--------------------- Binary Encoding ------------------------#
    
    cat_cols = dataset.select_dtypes(include=['category']).columns.tolist()
    
    encoder = ce.BinaryEncoder(cols= cat_cols)
    
    dataset = encoder.fit_transform(dataset)
    
    return dataset

#________________________________main_____________________________
    
trainingDataSet= pd.read_csv('train_set.csv')

trainingDataSet.drop('Id',axis=1,inplace=True)

ntrain_row = trainingDataSet.shape[0]



y=trainingDataSet[['SalePrice']]
trainingDataSet.drop('SalePrice',axis=1,inplace=True)

#_____________________________test________________________

testDataSet= pd.read_csv('test.csv')
ntest_row = testDataSet.shape[0]


id_test=testDataSet['Id']
testDataSet.drop('Id',axis=1,inplace=True)

features = pd.concat((trainingDataSet, testDataSet), sort=False).reset_index(drop=True)

proccessed_data=processData(features)

X=proccessed_data[:ntrain_row]
x_test_toPredict=proccessed_data[ntrain_row:]


#------------------------------------------
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

#-------------------------------------------------------------------------------------
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse
        
#------------------  creat the regression model ----------------------------------------#

regressor = linear_model.LinearRegression()

def adj_r2(x,y):
    r2 = regressor.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

regressor.fit(x_train,y_train)        
y_predect=regressor.predict(x_test_toPredict)

print('linear score R2',regressor.score(x_test,y_test))
print('linear score_AdjR2',adj_r2(x_test,y_test))

score = rmsle_cv(regressor)
print("\nlinear regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))





#----------------- Ridge regression   ----------------------------------------#

#------------ Hyperparameter(Alpha) tuning with GridSearchCV  --------------#
# prepare a range of alpha values to test
alphas_ridge = np.logspace(-4, 0, 50)
# create and fit a lasso regression model, testing each alpha
ridge=Ridge()
ridge_cv = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas_ridge))
ridge_cv.fit(x_train,y_train)
ridge_best_param = ridge_cv.best_params_
ridge_best_score = ridge_cv.best_score_

#----------------------------------------''--------------------#
ridge = Ridge(alpha=ridge_best_param['alpha'], normalize=True)
ridge.fit(x_train,y_train)
ridge_pred = ridge.predict(x_test)
final_pridect=ridge.predict(x_test_toPredict)

plt.scatter(y_test,ridge_pred)

print('Ridge score:',ridge.score(x_test, y_test))

#---------------- cv_scores---------------#
cv_scores_ridge = cross_val_score(ridge, x_train,y_train, cv=5)
print("Average 5-Fold CV Score_ridge: {}".format(np.mean(cv_scores_ridge)))
plt.clf

to_file=pd.DataFrame({'Id':id_test})  
to_file['SalePrice']=final_pridect
to_file.to_csv('omnia_hala.csv',index=False)
