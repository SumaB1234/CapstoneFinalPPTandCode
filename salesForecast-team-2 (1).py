#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip install xgboost')
get_ipython().system('pip install plotly')
get_ipython().system('pip install seaborn')



#importing numpy
import numpy as np 
#importing pandas
import pandas as pd 
#importing matplotlib
import matplotlib.pyplot as plt
#importing plotly
import plotly.express as px
#importing seaborn
import seaborn


train_dataframe = pd.read_csv('./train.csv', usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'])
test_dataframe = pd.read_csv('./test.csv',usecols=['store_nbr', 'family', 'date', 'onpromotion'])



train_dataframe.head()
#converting store numbers to string
train_dataframe['store_nbr'] = train_dataframe['store_nbr'].astype('category')
test_dataframe['store_nbr'] = train_dataframe['store_nbr'].astype('category')
#converting date to date type
train_dataframe['date'] = pd.to_datetime(train_dataframe['date'])
test_dataframe['date'] = pd.to_datetime(test_dataframe['date'])
#extracting year from date
train_dataframe['year'] = train_dataframe.date.dt.year
test_dataframe['year'] = test_dataframe.date.dt.year
#extracting month from date
train_dataframe['month'] = train_dataframe.date.dt.month
test_dataframe['month'] = test_dataframe.date.dt.month
#extracting day of month from date
train_dataframe['dayofmonth'] = train_dataframe.date.dt.day
test_dataframe['dayofmonth'] = test_dataframe.date.dt.day
#extracting day of week from date
train_dataframe['dayofweek'] = train_dataframe.date.dt.dayofweek
test_dataframe['dayofweek'] = test_dataframe.date.dt.dayofweek
#extracting day name from date
train_dataframe['dayname'] = train_dataframe.date.dt.strftime('%A')
test_dataframe['dayname'] = test_dataframe.date.dt.strftime('%A')


#printing minimum and maximum
print('minimum : ',min(train_dataframe.date),max(train_dataframe.date),'\n')
print('maximum: ',min(test_dataframe.date),max(test_dataframe.date))
train_dataframe.shape
#printing any missing values
print('Missing values in train:', train_dataframe.isna().sum().sum())
print('Missing values in test:', test_dataframe.isna().sum().sum())

#printing last 20 values.
train_dataframe.tail(20)

#grouping the values by store ID's
temp = train_dataframe.set_index('date').groupby('store_nbr').resample('D').sales.sum().reset_index()
#ploting the graph
px.line(temp, x='date', y='sales', color = "store_nbr",
        title='Daily total sales of the stores')


train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '52') & (train_dataframe.date < "2017-04-20"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '22') & (train_dataframe.date < "2015-10-09"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '42') & (train_dataframe.date < "2015-08-21"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '21') & (train_dataframe.date < "2015-07-24"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '29') & (train_dataframe.date < "2015-03-20"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '20') & (train_dataframe.date < "2015-02-13"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '53') & (train_dataframe.date < "2014-05-29"))]
train_dataframe = train_dataframe[~((train_dataframe.store_nbr == '36') & (train_dataframe.date < "2013-05-09"))]
train_dataframe.shape

#correlation
train_dataframe.corr('spearman').sales.loc['onpromotion']

#grouping store numbers by onpromotions and sales and ploting it.
train_dataframe.groupby('store_nbr')[['onpromotion','sales']].sum().plot.scatter('onpromotion','sales')
plt.title('Promotion and Sales Relationship')

#week names
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
#grouping by daymanes and ploting them
train_dataframe.groupby('dayname').sales.mean().reindex(index=order).plot(kind='bar')
plt.title('Average Sales by Day of week')

#reading transactions file
transaction_df = pd.read_csv('./transactions.csv', dtype={'store_nbr': 'category'})
#printing first 5 values.
transaction_df.head()
#converting string into date format
transaction_df['date'] = pd.to_datetime(transaction_df['date'])

# Proof that transactions are highly correlated with sales
temp = pd.merge(train_dataframe.groupby(['date', 'store_nbr']).sales.sum().reset_index(),transaction_df, how='left')
print(temp.corr("spearman").sales.loc["transactions"])

# Now we can proof that stores on holidays make more money than on working days
temp = transaction_df.copy()
#extracting year from date
temp['year'] = temp.date.dt.year
#extracting day of week from date
temp['dayofweek'] = temp.date.dt.dayofweek + 1
temp = temp.groupby(['year', 'dayofweek']).transactions.mean().reset_index()

#ploting the transactions
px.line(temp, x='dayofweek', y='transactions', color='year', title='Transactions')
#reading stores file
stores = pd.read_csv('./stores.csv',index_col='store_nbr')
#merging train and stores
train_dataframe = pd.merge(train_dataframe,stores,how='left',on='store_nbr')
#merging test and stores
test_dataframe = pd.merge(test_dataframe,stores,how='left',on='store_nbr')
#ploting sales 
train_dataframe.groupby(['type']).sales.mean().plot(kind='bar');

#figure size
plt.figure(figsize=(10,4))
#ploting the values
ax1 = plt.subplot(1,2,1)
train_dataframe.groupby(['city']).sales.mean().plot(kind='bar')
plt.title('Average Sales by City')
ax2 = plt.subplot(1,2,2)
train_dataframe.groupby(['city'])['store_nbr'].nunique().plot(kind='bar')
plt.title('Number of Stores by City')

#reading the holidays events
holiday_df = pd.read_csv('./holidays_events.csv')
#query
holiday_df.query('transferred==True')

# transferred day is not celebrated
holiday_df = holiday_df.query('transferred ==False')
holiday_df.description = holiday_df.description.str.replace('Traslado ','')

#national
national = holiday_df.query('locale=="National"')
#checking about the day
day_off = national.query('type!="Work Day" or type!="Event"').set_index('date')['description'].to_dict()
#converting date into date format
train_dataframe['date_str'] = train_dataframe.date.astype(str)
test_dataframe['date_str'] = test_dataframe.date.astype(str)
#checking weather the is national holiday or not.
train_dataframe['national_holiday'] = [1 if a in day_off else 0 for a in train_dataframe.date_str]
test_dataframe['national_holiday'] = [1 if a in day_off else 0 for a in test_dataframe.date_str]

event = national.query('type=="Event"').set_index('date')['description'].to_dict()
#checking weather the is national event or not.
train_dataframe['national_event'] =[1 if a in event else 0 for a in train_dataframe.date_str]
test_dataframe['national_event'] =[1 if a in event else 0 for a in test_dataframe.date_str]
#checking weather the is national workday or not.
work_day = national.query('type=="Work Day"').set_index('date')['description'].to_dict()
train_dataframe['national_workday'] = [1 if a in work_day else 0 for a in train_dataframe.date_str]
test_dataframe['national_workday'] = [1 if a in work_day else 0 for a in test_dataframe.date_str]

#checking weather the is weekend or not.
train_dataframe['weekend'] = [1 if a>=5 else 0 for a in train_dataframe.dayofweek]
test_dataframe['weekend'] = [1 if a>=5 else 0 for a in test_dataframe.dayofweek]

#checking weather the is nlocal holiday or not.
local = holiday_df.query('locale=="Local"')
local_dic = local.set_index('date').locale_name.to_dict()
train_dataframe['local_holiday']=[1 if b in local_dic and local_dic[b]== a else 0 for a,b in zip(train_dataframe.city,train_dataframe.date_str)]
test_dataframe['local_holiday']=[1 if b in local_dic and local_dic[b]== a else 0 for a,b in zip(test_dataframe.city,test_dataframe.date_str)]


#checking weather the is regional holiday or not.
regional = holiday_df.query('locale=="Regional"')
regional_dic = regional.set_index('date').locale_name.to_dict()
train_dataframe['regional_holiday']= [1 if b in regional_dic and regional_dic[b]== a else 0 for a,b in zip(train_dataframe.state,train_dataframe.date_str)]
test_dataframe['regional_holiday']= [1 if b in regional_dic and regional_dic[b]== a else 0 for a,b in zip(test_dataframe.state,test_dataframe.date_str)]

#reading oil file
oil = pd.read_csv('./oil.csv')
oil['date'] = pd.to_datetime(oil['date'])
#printing firsgt 5 values
oil.head()


# sampling again
oil = oil.set_index('date')['dcoilwtico'].resample(
    'D').sum().reset_index()  # add missing dates and fill NaNs with 0 

# interpolating the values to fill the null values.
oil['dcoilwtico'] = np.where(oil['dcoilwtico']==0, np.nan, oil['dcoilwtico'])  # replace 0 with NaN
oil['dcoilwtico_interpolated'] = oil.dcoilwtico.interpolate()  # fill NaN values using an interpolation method
#printing first 5 values.
oil.head(10)

temp = oil.melt(id_vars=['date'], var_name='Legend') 
px.line(temp.sort_values(['Legend', 'date'], ascending=[False, True]), x='date',
        y='value', color='Legend', title='Daily Oil Price')



import matplotlib.pyplot as plt
#sale and oil price dependency function.
def plot_sales_and_oil_dependency():
    a = pd.merge(train_dataframe.groupby(["date", "family"]).sales.sum().reset_index(),
                 oil.drop("dcoilwtico", axis=1), how="left")
    c = a.groupby("family").corr("spearman").reset_index()
    c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")
    
    fig, axes = plt.subplots(7, 5, figsize=(20, 20))
    for i, fam in enumerate(c.family):
        a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=axes[i // 5, i % 5])
        axes[i // 5, i % 5].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6],
                                 fontsize=12)
        axes[i // 5, i % 5].axvline(x=70, color='r', linestyle='--')

    plt.tight_layout(pad=5)
    plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize=20)
    plt.show()

plot_sales_and_oil_dependency()


oil = oil.set_index('date').dcoilwtico.interpolate(method='linear').to_frame()
oil['date_str'] = oil.index.astype(str)



train_dataframe['date_str'] = train_dataframe.date.astype(str)
train_dataframe = pd.merge(train_dataframe,oil,how='left',on='date_str')


test_dataframe['date_str'] = test_dataframe.date.astype(str)
test_dataframe = pd.merge(test_dataframe,oil,how='left', on='date_str')


len(train_dataframe.query('date_str=="2013-01-01"'))



train_dataframe.sales = np.log1p(train_dataframe.sales)



train_dataframe['Istest'] = False
test_dataframe['Istest'] = True

full = pd.concat((train_dataframe,test_dataframe))

#remove leap year day
#full = full.query('date_str !="2016-02-29"')


full['Lag_16'] = full['sales'].shift(1782*16)
full['Lag_17'] = full['sales'].shift(1782*17)
full['Lag_18'] = full['sales'].shift(1782*18)
full['Lag_19'] = full['sales'].shift(1782*19)
full['Lag_20'] = full['sales'].shift(1782*20)
full["Lag_21"] = full['sales'].shift(1782*21)
full['Lag_22'] = full['sales'].shift(1782*22)
full['Lag_28'] = full['sales'].shift(1782*28)
full['Lag_31'] = full['sales'].shift(1782*31)

full['Lag_365'] = full['sales'].shift(1782*365)


train_dataframe = full.query('Istest==False')
test_dataframe = full.query('Istest ==True')



train_dataframe = train_dataframe.dropna(subset=['Lag_365'],axis=0)


#list of features
FEATURES = ['store_nbr','family','onpromotion', 'year', 'month',
       'dayofmonth', 'dayofweek','dcoilwtico', 'city', 'state',
       'type', 'cluster', 'national_holiday', 'national_event',
       'national_workday', 'weekend', 'local_holiday', 'regional_holiday','Lag_16','Lag_17','Lag_18','Lag_19','Lag_20','Lag_21','Lag_22','Lag_28','Lag_31','Lag_365']
TARGET =['sales']


#importing the skleaern
from sklearn import preprocessing
categories = ['family','city','state','type']
for i in categories:
    #preprocessing by using label encoder.
    encoder = preprocessing.LabelEncoder()
    train_dataframe[i] = encoder.fit_transform(train_dataframe[i])
    test_dataframe[i] =  encoder.transform(test_dataframe[i])


#importing train test and split
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(train_dataframe,train_dataframe[TARGET],test_size=0.05,shuffle=False)


#importing linear regression from sklearn
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_log_error

linear = LinearRegression()
#fitting the model
model = linear.fit(X_train[FEATURES],y_train)


#predicting the values of linear regression
predictions= model.predict(X_val[FEATURES])
predictions = [a if a>0 else 0 for a in predictions]
#calculating the mean_squared_log_error
print('MSLE: ' + str(mean_squared_log_error(y_val,predictions)))


#importing XGBoost
from xgboost import XGBRegressor
#fitting the model into the model
xgb = XGBRegressor(n_estimators=500)
xgb.fit(X_train[FEATURES], y_train,
        eval_set=[(X_train[FEATURES],y_train),(X_val[FEATURES], y_val)],
       verbose=False,early_stopping_rounds=10)

#predecting the values
predictions= xgb.predict(X_val[FEATURES])
predictions = [a if a>0 else 0 for a in predictions]
#calculating the mean_squared_log_error
print('MSLE: ',mean_squared_log_error(y_val,predictions))

