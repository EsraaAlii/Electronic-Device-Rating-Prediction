#!/usr/bin/env python
# coding: utf-8

# In[1890]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing  import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("ElecDeviceRatingPrediction.csv")
electro = df.copy()
electro.head()


# In[1891]:


#check the datatypes
print(electro.dtypes)


# In[1892]:


electro['ram_gb']=electro['ram_gb'].str.strip('GB')


# In[1893]:


meanValue_ratings = electro['Number of Ratings'].mean()
electro['Number of Ratings']=electro['Number of Ratings'].replace(0, meanValue_ratings)
electro['Number of Ratings']=electro['Number of Ratings'].astype('int')


# In[1894]:


electro['reviews per rating']= electro['Number of Reviews']/electro['Number of Ratings']


# In[1895]:


electro['reviews - rating']= electro['Number of Reviews']-electro['Number of Ratings']


# In[1896]:


electro['reviews + rating']= electro['Number of Reviews']+electro['Number of Ratings']


# In[1897]:


electro['reviews * rating']= electro['Number of Reviews']*electro['Number of Ratings']


# In[1898]:


electro


# In[1899]:


print(electro.dtypes)


# In[1900]:


mode_value = electro['processor_gnrtn'].mode()[0]

print(mode_value)


# In[1901]:


#convert to the correct datatype
electro['Touchscreen']=electro['Touchscreen'].astype('category')
electro['msoffice']=electro['msoffice'].astype('category')
electro['rating']=electro['rating'].str.strip('stars')
electro['rating']=electro['rating'].astype('int')

electro['ram_gb']=electro['ram_gb'].astype('int')
electro['ssd']=electro['ssd'].str.strip('GB')
electro['ssd']=electro['ssd'].astype('int')
electro['hdd']=electro['hdd'].str.strip('GB')
electro['hdd']=electro['hdd'].astype('int')
electro['processor_brand']=electro['processor_brand'].astype('category')
electro['brand']=electro['brand'].astype('category')
electro['processor_name']=electro['processor_name'].astype('category')
electro['ram_type']=electro['ram_type'].astype('category')
electro['os']=electro['os'].astype('category')
electro['weight']=electro['weight'].astype('category')
electro['graphic_card_gb']=electro['graphic_card_gb'].str.strip('GB')
electro['graphic_card_gb']=electro['graphic_card_gb'].astype('int')
electro['warranty']=electro['warranty'].str.strip('year || years')
electro['warranty']=electro['warranty'].str.replace('No warrant','0')
electro['warranty']=electro['warranty'].astype('int')
electro['processor_gnrtn']=electro['processor_gnrtn'].str.replace('Not Available',mode_value)
electro['processor_gnrtn']=electro['processor_gnrtn'].str.strip('th')


# In[1902]:


print('unique values of ram type : ',electro['ram_type'].unique())


# In[1903]:


#manually encoding ram type
manual_encoding = {}
for category in electro['ram_type'].unique():
    if(category == 'DDR5'):
        manual_encoding[category] = 5
    elif(category == 'LPDDR4X'):
        manual_encoding[category] = 4
    elif(category == 'LPDDR4'):
        manual_encoding[category] = 3
    elif(category == 'DDR4'):
        manual_encoding[category] = 2
    elif(category == 'LPDDR3'):
        manual_encoding[category] = 1
    else:
        manual_encoding[category] = 0

print(manual_encoding)


# In[1904]:


electro.loc[electro['processor_gnrtn'] == '0' , 'processor_gnrtn'] = electro['processor_gnrtn'].value_counts().idxmax()
electro['processor_gnrtn']=electro['processor_gnrtn'].astype('int')


# In[1905]:


print('unique values of processor_gnrtn : ',electro['processor_gnrtn'].unique())


# In[1906]:


#check the ranges
print(electro.describe())


# In[1907]:


#check for duplicates
duplicates = electro.duplicated(keep=False)
print('no. of duplicates : ',duplicates.sum())


# In[1908]:


#drop the duplicates
electro.drop_duplicates(inplace=True)


# In[1909]:


#check if they are deleted
duplicates = electro.duplicated(keep=False).sum()
print('no. of duplicates : ',duplicates)


# In[1910]:


#check if there is any device doesn't have any processor
print(electro[(electro['ssd']==0) & (electro['hdd']==0)])


# In[1911]:


#check missing data
print('NULLS : \n',electro.isna().sum())


# In[1912]:


print('data types : \n',electro.dtypes)


# In[1913]:


#label encoding for features with natural order 
label_encoder = LabelEncoder()  
electro['weight']= label_encoder.fit_transform(electro['weight'])
electro['msoffice'] = label_encoder.fit_transform(electro['msoffice'])
electro['ram_type'] = label_encoder.fit_transform(electro['ram_type'])


# In[1914]:


categoricalData = pd.DataFrame()
categoricalData['weight'] = electro['weight']
categoricalData['brand'] = electro['brand']
categoricalData['processor_brand'] = electro['processor_brand']
categoricalData['processor_name'] = electro['processor_name']
categoricalData['os'] = electro['os']
categoricalData['Touchscreen'] = electro['Touchscreen']
categoricalData['msoffice'] = electro['msoffice']
categoricalData['ram_type'] = electro['ram_type']
#one hot encoding
categoricalData = pd.get_dummies(categoricalData,columns=['brand','processor_brand','processor_name','os','Touchscreen'],dtype=int)


# In[1915]:


#one hot encoding for categories with no order
electro=pd.get_dummies(electro,columns=['brand','processor_brand','processor_name','os','Touchscreen'],dtype=int)


# In[1916]:


#check the outliers
fig, axes = plt.subplots(4,2,sharex=True,figsize=(10, 10))
fig.suptitle('Check the outliers of numeric columns')
plt.subplots_adjust(wspace = 0.5)
sns.boxplot(data=electro, y='ram_gb', ax=axes[0,0])
sns.boxplot(data=electro, y='ssd', ax=axes[0,1])
sns.boxplot(data=electro, y='hdd', ax=axes[1,0])
sns.boxplot(data=electro, y='Number of Reviews', ax=axes[1,1])
sns.boxplot(data=electro, y='Number of Ratings', ax=axes[2,0])
sns.boxplot(data=electro, y='Price', ax=axes[2,1])
sns.boxplot(data=electro, y='warranty', ax=axes[3,0])
sns.boxplot(data=electro, y='graphic_card_gb', ax=axes[3,1])
plt.show()


# In[1917]:


#drop the outliers
for col in electro.columns:
        #print("capping the ",col)
        if (((electro[col].dtype)=='float') | ((electro[col].dtype)=='int32')|((electro[col].dtype)=='int64')):
            percentiles = electro[col].quantile([0.25,0.75]).values
            iqr= percentiles[1]-percentiles[0]
            electro[col][electro[col] <= percentiles[0]-1.5*iqr] = percentiles[0]
            electro[col][electro[col] >= percentiles[1]+1.5*iqr] = percentiles[1]
        else:
            electro[col]=electro[col]


# In[1918]:


#check the outliers
fig, axes = plt.subplots(4,2,sharex=True,figsize=(10, 10))
fig.suptitle('Check the outliers of numeric columns')
plt.subplots_adjust(wspace = 0.5)
sns.boxplot(data=electro, y='ram_gb', ax=axes[0,0])
sns.boxplot(data=electro, y='ssd', ax=axes[0,1])
sns.boxplot(data=electro, y='hdd', ax=axes[1,0])
sns.boxplot(data=electro, y='Number of Reviews', ax=axes[1,1])
sns.boxplot(data=electro, y='Number of Ratings', ax=axes[2,0])
sns.boxplot(data=electro, y='Price', ax=axes[2,1])
sns.boxplot(data=electro, y='warranty', ax=axes[3,0])
sns.boxplot(data=electro, y='graphic_card_gb', ax=axes[3,1])
plt.show()


# In[1920]:


#split to features and result
X = electro.drop(columns=['rating'])
Y = electro['rating']


# In[1921]:


Y_categorical = electro['rating']
scaler =  MinMaxScaler(feature_range=(0,1))
X= pd.DataFrame(scaler.fit_transform(X),index=X.index,columns= X.columns)


# In[1922]:


print('unique values of rating : ',electro['rating'].unique())


# In[1923]:


#check the best features with Y
corr = electro.corr()
top_feature = corr.loc[abs(corr['rating'])>0.15,'rating']
top_feature=pd.DataFrame(top_feature)
sns.heatmap(top_feature,annot=True)
plt.show()
top_feature


# In[1925]:


Y=electro['rating']
top_feature = top_feature.drop('rating')

#top_feature
first_X= X[top_feature.index]


# In[1926]:


best_features = SelectKBest(score_func=chi2, k=3)
fit = best_features.fit(categoricalData, Y_categorical)

# Summarize scores
chi_scores = pd.DataFrame(fit.scores_, columns=["Chi-Square Score"])
columns = pd.DataFrame(categoricalData.columns, columns=["Feature"])
chi_summary = pd.concat([columns, chi_scores], axis=1)

print("Chi-Square scores:\n", chi_summary)


# In[1927]:


# Calculate information gain for each feature
info_gain = mutual_info_regression(X, Y)

# Select features with information gain greater than 0.15
top_feature_idx = np.where(info_gain > 0.15)[0]
top_feature_names = X.columns[top_feature_idx]
top_feature_df = pd.DataFrame(data=info_gain[top_feature_idx], index=top_feature_names, columns=['Information Gain'])

plt.figure(figsize=(12, 6))
plt.bar(X.columns, info_gain,width=0.5)
plt.xticks(rotation = 90)
plt.xlabel('Features')
plt.ylabel('Information Gain')
plt.title('Information Gain for Feature Selection')
plt.show()
print('Top features of info gain',top_feature_df)
third_feature_selection = X[top_feature_df.index]
print('info gain features : \n',third_feature_selection)


# In[1928]:


#linear regression model
X_train, X_test, y_train, y_test = train_test_split(first_X, Y, test_size=0.25, random_state= 104, shuffle = True)



model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
y_pred_train = model.predict(X_train)
mse = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)
r2 = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)
print("Mean Squared Error (linear regression):", mse)
print("Mean Squared Error of train (linear regression):", mse_train)

print("R^2 Score (linear regression):", r2)
print("R^2 Score train (linear regression):", r2_train)


cv_mse = -cross_val_score(model, first_X, Y, cv=10, scoring='neg_mean_squared_error')
print("Cross-Validation Mean Squared Error (linear regression):", cv_mse.mean())

cv = KFold(n_splits=10, random_state=1, shuffle=True)

scores = cross_val_score(model, first_X, Y, 
                         cv=cv)

print("Accuracy (linear regression): " ,scores.mean())


# In[1929]:


#ridge regression model
model = Ridge(alpha=10)
model.fit(X_train, y_train)
y_ridge = model.predict(X_test)
y_ridge_train = model.predict(X_train)
mse_ridge = mean_squared_error(y_test, y_ridge)
mse_ridge_train = mean_squared_error(y_train, y_ridge_train)

r2_ridge = r2_score(y_test, y_ridge)


print("Mean Squared Error (ridge regression):", mse_ridge)
print("Mean Squared Error of train (ridge regression):", mse_ridge_train)
print("R^2 Score (ridge regression):", r2_ridge)

cv_mse_ridge = -cross_val_score(model, first_X, Y, cv=10, scoring='neg_mean_squared_error')
print("Cross-Validation Mean Squared Error (ridge regression):", cv_mse_ridge.mean())

cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, first_X, Y, 
                         cv=cv)

print("Accuracy (ridge regression): " ,scores.mean())


# In[1930]:


#lasso regression model
model = Lasso(alpha=0.001)
model.fit(X_train, y_train)
y_lasso = model.predict(X_test)
y_lasso_train = model.predict(X_train)

mse_lasso = mean_squared_error(y_test, y_lasso)
mse_lasso_train = mean_squared_error(y_train, y_lasso_train)

r2_lasso = r2_score(y_test, y_lasso)


print("Mean Squared Error (lasso regression):", mse_lasso)
print("Mean Squared Error of train (lasso regression):", mse_lasso_train)
print("R^2 Score (lasso regression):", r2_lasso)

cv_mse_lasso = -cross_val_score(model, first_X, Y, cv=10, scoring='neg_mean_squared_error')
print("Cross-Validation Mean Squared Error (lasso regression):", cv_mse_lasso.mean())

cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, first_X, Y, 
                         cv=cv)

print("Accuracy (lasso regression) : " ,scores.mean())


# In[1931]:


regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)

regressor.fit(X_train, y_train)


prediction_GB = regressor.predict(X_test)
prediction_train_GB = model.predict(X_train)
mse_GB = mean_squared_error(y_test, prediction_GB)
mse_GB_train = mean_squared_error(y_train, prediction_train_GB)

r2_GB = r2_score(y_test, prediction_GB)
r2_train_GB = r2_score(y_train, prediction_train_GB)



mse_GB = cross_val_score(regressor, first_X, Y, cv=10, scoring='neg_mean_squared_error')




print("Mean Squared Error of train (GradientBoost):", mse_GB_train)
print("R^2 Score (GradientBoost) :", r2_GB)
print("R^2 Score train (GradientBoost):", r2_train_GB)





mse_mean_GB = -mse_GB.mean()
print("Mean Squared Error (GradientBoost):", mse_mean_GB)
cv_GB = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(regressor, first_X, Y, 
                         cv=cv_GB, scoring ='r2')

print("Accuracy (GradientBoost) : " ,scores.mean())


# In[1932]:


regressor = RandomForestRegressor(n_estimators=100, random_state=5)


regressor.fit(X_train, y_train)


prediction_RF = regressor.predict(X_test)

prediction_train_RF = model.predict(X_train)
mse_RF = mean_squared_error(y_test, prediction_RF)
mse_RF_train = mean_squared_error(y_train, prediction_train_RF)

r2_RF = r2_score(y_test, prediction_RF)
r2_train_RF = r2_score(y_train, prediction_train_RF)

cv_mse_RF = -cross_val_score(regressor, first_X, Y, cv=10, scoring='neg_mean_squared_error')
print("Cross-Validation Mean Squared Error (RandomForest):", cv_mse_RF.mean())



print("Mean Squared Error (RandomForest):", mse_RF)
print("Mean Squared Error of train (RandomForest):", mse_RF_train)
print("R^2 Score (RandomForest):", r2_RF)
print("R^2 Score train (RandomForest):", r2_train_RF)


cv_RF = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(regressor, first_X, Y, 
                         cv=cv_RF, scoring ='r2')

print("Accuracy (RandomForest): " ,scores.mean())


# In[1933]:


#polynomial regression
polynomial_features = PolynomialFeatures(degree=2)
first_X_poly = polynomial_features.fit_transform(X_train)
first_X_poly_test = polynomial_features.fit_transform(X_test)


model = LinearRegression()
model.fit(first_X_poly, y_train)
y_poly_pred = model.predict(first_X_poly_test)
y_poly_train = model.predict(first_X_poly)
mse_poly = mean_squared_error(y_test, y_poly_pred)
mse_poly_train = mean_squared_error(y_train, y_poly_train)

r2_poly = r2_score(y_test, y_poly_pred)


print("Mean Squared Error (polynomial regression):", mse_poly)
print("Mean Squared Error of train (polynomial regression):", mse_poly_train)
print("R^2 Score (polynomial regression):", r2_lasso)

cv_mse_poly = -cross_val_score(model, first_X, Y, cv=10, scoring='neg_mean_squared_error')
print("Cross-Validation Mean Squared Error:", cv_mse_poly.mean())

cv_poly = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, first_X, Y, 
                         cv=cv_poly)

print("Accuracy : " ,scores.mean())

