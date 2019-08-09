# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:07:51 2019

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from scipy.stats import johnsonsu,boxcox_normmax
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, accuracy_score,mean_squared_error





#----------------  analysis of general characteristics of the data.  --------------------------------------------
    
def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str

#===============================================================================================

#------------------Target Variables ------------------------------------------------------------------------#
def Target_Variable_look(df_dataset):
    cols_name=[]
    cols_name = df_dataset.columns.values.tolist()
 # --------------- #--Let's look at the distribution of the Target Variable----------------#
    plt_t_variable = plt.figure()
    sns.distplot(df_dataset['price'] , fit=norm);
    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df_dataset['price'])
        #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
    plt.ylabel('Frequency')
    plt.title('Car price distribution')
    plt.clf
    plt_t_variable.show()
    return 



#----------- Log-transform target var -----------------------------------#
def Best_log_target_variable(train) : # It takes Dataframe
    y = train['price']
    
    plt.figure_1= plt.figure()
    sns.distplot(y, kde=False, fit=stats.johnsonsu)
    plt.title('Johnson SU')
    plt.figure_1.show()
    plt.clf
    
    plt.figure_2= plt.figure()
    sns.distplot(y, kde=False, fit=stats.norm)
    plt.title('Normal')
    plt.figure_2.show()
    plt.clf
   
    plt.figure_3= plt.figure()
    sns.distplot(y, kde=False, fit=stats.lognorm)
    plt.title('Log Normal')
    plt.figure_3.show()
    plt.clf

    return 



# #------------------Target Variables --------------------------#
def Log_Target_Variable(df_dataset):
 # --------------- #--Let's look at the distribution of the Target Variable----------------#
 #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df_dataset['price'])
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    df_dataset["price"] = np.log1p(df_dataset["price"])
    
    #Check the new distribution
    plt.figure_1= plt.figure()
    sns.distplot(df_dataset['price'] , fit=stats.johnsonsu);
    
    #Now plot the distribution
    plt.legend(['johnsonsu dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
    plt.ylabel('Frequency')
    plt.title('Car price distribution - Johnson SU')
    plt.figure_1.show()
    plt.clf
    return


#----------------- Heat map ----------------#
    
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'price']].groupby(feature).mean()['price']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    



#------------------
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['price'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
#-------------------Have_To_Log ------------------------#
def Have_To_Log (train):
    test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
    normal = pd.DataFrame(train[quantitative_processed])
    normal = normal.apply(test_normality)
    print(not normal.any()) # if the resulte is FALSE mean that all columns have to log
    return 
    



#--------------------------- Log/transform all quantitative feature NOT nrmaly distrubuted-----------------------------------------------------------#
#We transform this variable and make it more normally distributed.
#We use a log1p transformation: np.log1p = log(1+p)
#We do this to speed up the learning and convergence time of Linear ML Models
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
def log_col (df_y_target):
    cols_name=[]
    cols_name = df_y_target.columns.values.tolist()
    plt_log = plt.figure()
    (mu, sigma) = norm.fit(df_y_target[cols_name[0]])
    df_y_target[cols_name[0]] = np.log1p(df_y_target[cols_name[0]])
        #Check the new distribution 
    sns.distplot(df_y_target[cols_name[0]].dropna() , fit=stats.johnsonsu);
        #Now plot the distribution
    plt.legend(['Johnson SU. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
    plt.ylabel('Frequency')
    plt.title(cols_name[0]+' distribution')
    plt_log.show()
    return
#======================================================================================================

#------------------ MAIN ------------------------------------#
#--------------------------   Load data    -----------------------------------------------------#
#df= pd.read_csv('imports_85_data 2.csv')

df= pd.read_csv('imports_85_data 2.csv')

df.columns = ['symboling','normalized_losses','make','fuel_type','aspiration',
              'num_of_doors','body_style','drive_wheels','engine_location','wheel_base',
              'length','width','height','curb_weight','engine_type','num_of_cylinders','engine_size',
              'fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm',
              'city_mpg','highway_mpg','price']

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object'] 
  
#==================================================
    
details = rstr(df,'price')


#-------------------- Heat map -----------------------#
df_heatmap = df.copy()
qual_encoded = []
for q in qualitative:  
    encode(df_heatmap, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)
#-----------------
features = quantitative + qual_encoded
#-----------------
plt.figure(1)
corr = df[quantitative+['price']].corr()
sns.heatmap(corr)
plt.figure(2)
corr = df_heatmap[qual_encoded+['price']].corr()
sns.heatmap(corr)
plt.figure(3)
corr = pd.DataFrame(np.zeros([len(quantitative)+1, len(qual_encoded)+1]), index=quantitative+['price'], columns=qual_encoded+['price'])
for q1 in quantitative+['price']:
    for q2 in qual_encoded+['price']:
        corr.loc[q1, q2] = df_heatmap[q1].corr(df_heatmap[q2])
sns.heatmap(corr)

#-------------- missing value target variable----------------#
df = df.dropna(how='all') #Drop the rows if entire row has NaN (missing) values
df = df.dropna(subset=['price']) #Drop rows with NaN in a specific column 'price'


# =============================================================================
# # #----------------    Target Variables   ---------------------------------------------------------------------## ------------Let's look at the distribution of the Target Variable------#
# Target_Variable_look(df) # It has to log based on the plot 
# 
# 
# Best_log_target_variable(df) # fit with Johnson SU
# 
# Log_Target_Variable(df)
# =============================================================================

#------------------ Data processing  -------------------------------------#

    #--------  change num of doors type -----------#
df['num_of_doors'] = df.num_of_doors.map({'four':4,'two':2, '?':4})
#--------  change num_of_cylinders type -----------#
df['num_of_cylinders'] = df.num_of_cylinders.map({'four':4,'six':6, 'five':5,'eight':8,'two':2,'twelve':12,'three':3})

#Changing symboling into a categorical variable
df['num_of_cylinders'] = df['num_of_cylinders'].astype(str)

#-----------  symboling -------------------#
df['symboling'] = df.symboling.map({3:0, 2:1, 1:2, 0:3,-1:4,-2:5})
#Changing symboling into a categorical variable
df['symboling'] = df['symboling'].astype(str)

    
#------------------ Missing values -------------------------------------#
#-------------- peak_rpm -------------------#
df["peak_rpm"] = df.groupby("symboling")["peak_rpm"].transform(lambda x: x.fillna(x.mean()))


#------------ hors power -----------------#
df["horsepower"] = df.groupby("num_of_cylinders")["horsepower"].transform(lambda x: x.fillna(x.median()))

df['horsepower'] = df['horsepower'].astype(str)

#---------- normalized_losses ------#
df['normalized_losses'] = df['normalized_losses'].fillna(df['normalized_losses'].mean())


#---------------- bore/stroke -------------#
#df['bore'] = df['bore'].fillna(df['bore'].mean())
#df['stroke'] = df['stroke'].fillna(df['stroke'].mean())


#---------------- variance / value_counts -------------#
df_var = df.var()
details = rstr(df,'price')
cols_obj = [f for f in df.columns if df.dtypes[f] != 'object']
for col in cols_obj:
    df[col].value_counts()
    #print(df[col].value_counts())
    
df = df.drop(['engine_location'], axis=1) # >90 same value
df = df.drop(['bore'], axis=1) # littil variance
df = df.drop(['stroke'], axis=1) # littil variance
#---------------------------------------------------#


# =============================================================================
# # -------------- log ---------------------#
# quantitative_processed = [f for f in df.columns if df.dtypes[f] != 'object']
# qualitative_processed = [f for f in df.columns if df.dtypes[f] == 'object'] 
# 
# Have_To_Log (df)
# 
# 
# df_quantitative_proc = pd.DataFrame(df[quantitative_processed])
# for i in df_quantitative_proc:
#     df_log = df[[i]]
#     log_col (df_log)
# 
# =============================================================================

#Have_To_Log (df)

#------------------ Heat map ------------------------------------------------#

plt.figure(figsize=(15, 15))
ax = sns.heatmap(df.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)
plt.title('Cross correlation between numerical')
plt.show()

#----------------- feature selection based on the corr between them - see heat map ------------------------#
## Above graph shows Wheel base , Length , Width are highly correlated. 
## Highway mpg and city mpg is also highly correlated. 
## Compression ratio and fuel type is also correlated 
## Engine size and horse power is also correlated
df = df.drop(['length','width','city_mpg','horsepower'],axis=1)


# =============================================================================
# #----------------------------  Skewed feauter Highest-------------------------------------------#
# quantitative_preceesed = [f for f in df.columns if df.dtypes[f] != 'object']
# skew_features = df[quantitative_preceesed].apply(lambda x: skew(x)).sort_values(ascending=False)
# 
# high_skew = skew_features[skew_features > 0.5]
# skew_index = high_skew.index
# 
# for i in skew_index:
#     df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))
# =============================================================================

# #---------------------- categoral data -------------------------------#

#  #-----------------------  change 'object'  column type to 'category' type --------#
df[df.select_dtypes(['object']).columns]=df.select_dtypes(include=['object']).apply(lambda x: x.astype('category'))
#     
#---------------------  label encoder --------------------#
labelencoder = LabelEncoder()
for i in ['fuel_type','aspiration', 'num_of_doors']:
    df[i] = labelencoder.fit_transform(df[i])


 #--------------------- Binary Encoding ------------------------#
#     
cat_cols = df.select_dtypes(include=['category']).columns.tolist()
encoder = ce.BinaryEncoder(cols= cat_cols)
df= encoder.fit_transform(df)


#------------------------declar X,y variables --------------------------------------------#

X = df.drop(['price'], axis=1)
y=df[['price']]
df.drop('price',axis=1,inplace=True)

# =============================================================================

# =============================================================================
# #---------------------------------- scaler --------------------------------------------------------#    
# X=X.select_dtypes(include=[np.number])
# scaler = StandardScaler()
# X= pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
# =============================================================================
#---------------  train_test_split  ----------#
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

 # function ----
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse
# =============================================================================
# #--------------------- Multiple Linear Regression-----------------------------#
from sklearn.metrics import r2_score
# 
# reg = LinearRegression()
# reg.fit(x_train, y_train)
# 
# #--- To see what your score is
# linear_y_pred = reg.predict(x_test)
# 
# print("\nLinear _r2_score\n",r2_score(y_test, linear_y_pred))
#     #-----------------#
# print("linear_score_R2 = " , reg.score(x_test,y_test))
# 
# # #---------------- cross_val_score---------------#
# cv_scores_reg = cross_val_score(reg, x_train,y_train, cv=5)
# print("linear_Average 5-Fold CV Score: {}".format(np.mean(cv_scores_reg)))
# 
# =============================================================================



#--------------------------  Lasso --------------------------------------------#

lasso = make_pipeline(Lasso(alpha =0.0005, random_state=1))
lasso.fit(x_train, y_train)
lasso_y_predt = lasso.predict(x_test)

print("lasso_score = " , lasso.score(x_test,y_test))

print("Lasso _r2_score",r2_score(y_test, lasso_y_predt))

# #---------------- cross_val_score---------------#
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
cv_scores_lasso = cross_val_score(lasso, x_train,y_train, cv=kf)
print("lasso_Average 5-Fold CV Score: {}".format(np.mean(cv_scores_lasso)))

#print("\nRandom_mean_squared_error (rmse) = \n",np.sqrt(mean_squared_error(y_test, lasso_y_predt)))

#combined rmse value
rss=((y_test-lasso_y_predt)**2).sum()
mse=np.mean((y_test-lasso_y_predt)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-lasso_y_predt)**2)))










# =============================================================================
# #----------------- Ridge regression   ----------------------------------------#
# 
# #------------ Hyperparameter(Alpha) tuning with GridSearchCV  --------------#
# # prepare a range of alpha values to test
# alphas_ridge = np.logspace(-4, 0, 50)
# # create and fit a lasso regression model, testing each alpha
# ridge=Ridge(normalize=True)
# ridge_cv = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas_ridge))
# ridge_cv.fit(x_train,y_train)
# ridge_best_param = ridge_cv.best_params_
# ridge_best_score = ridge_cv.best_score_
# 
# ridge_cv_pred = ridge_cv.predict(x_test)
# 
# print("Ridg_score_R2 = " , ridge_cv.score(x_test,y_test))
# 
# print("Ridg _ r2_score",r2_score(y_test, ridge_cv_pred)) 
# 
# # #---------------- cross_val_score---------------#
# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
# cv_scores_ridg = cross_val_score(ridg, x_train,y_train, cv=kf)
# print("ridg_Average 5-Fold CV Score: {}".format(np.mean(cv_scores_ridg)))
# # 
# print("\nRandom_mean_squared_error (rmse) = \n",np.sqrt(mean_squared_error(y_test,ridge_cv_pred)))
# # 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # =============================================================================
# # #------------------------- Decision Trees¶ -----------------------------------#
# # tree = make_pipeline(RobustScaler(), DecisionTreeRegressor())
# # tree.fit(x_train, y_train)
# # # Reverse the log transformation
# # tree_pred = np.expm1(tree.predict(x_test))
# # tree_pred
# # # To see what your score is
# # 
# # print('Decision Tree score R2:',tree.score(x_test, y_test))
# # score_tree = rmsle_cv(tree)
# # print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score_tree.mean(), score_tree.std()))
# # 
# # =============================================================================
# 
# 
# 
# 
# 
# 
# 
#  #-------------------------- Random Forest ------------------------------------#
# forest = make_pipeline(RobustScaler(), RandomForestRegressor(min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0, n_estimators=100, bootstrap=True))
# forest.fit(x_train, y_train)
# # Reverse the log transformation
# forest_pred = forest.predict(x_test)
# forest_pred
# 
# 
# # To see what your score is
# 
# print('\nRandom Forest score R2:\n',forest.score(x_test, y_test))
# print("\nRandom_mean_squared_error = \n",np.sqrt(mean_squared_error(y_test,forest_pred)))
# 
# 
# # #---------------- cv_scores---------------#
# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
# cv_scores_forest = cross_val_score(forest, x_train,y_train, cv=5)
# #print("Average 5-Fold CV Score_ridge: {}".format(np.mean(cv_scores_forest)))
# 
# print("Random Forest_cv_scores_ Accuracy: %0.2f (+/- %0.2f)" % (cv_scores_forest.mean(), cv_scores_forest.std() * 2))
# 
# # =============================================================================
# # #----------------------  Elastic Net -----------------------------------------#
# # from sklearn.linear_model import ElasticNet
# # ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0))
# # ENet.fit(x_train, y_train)
# # enet_pred = np.expm1(ENet.predict(x_test))
# # enet_pred
# # score_Elastic = rmsle_cv(ENet)
# # 
# # print('Elastic Net score R2:',ENet.score(x_test, y_test))
# # print("\nENet(scaled) score: {:.4f} ({:.4f})\n".format(score_Elastic.mean(), score_Elastic.std()))
# # 
# # =============================================================================
# 
# #--------------------------   Gradient Boosting  ---------------------------#
#  
# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=3,min_samples_split=10, 
#                                    loss='huber', random_state =5)
# 
# GBoost.fit(x_train, y_train)
# GBoost_pred = np.expm1(GBoost.predict(x_test))
# GBoost_pred
# print('Gradient Boosting score R2:',GBoost.score(x_test, y_test))
# score_Gradient = rmsle_cv(GBoost)
# print("\nGBoost score: {:.4f} ({:.4f})\n".format(score_Gradient.mean(), score_Gradient.std()))
# =============================================================================
