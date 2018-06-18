import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url = 'C:\\Users\\Icheme\\Desktop\\ML\\Auto.txt'
Data = pd.read_csv(url, header = None)
print(Data)
headers = ['symboling','normalized-loss','make','fuel-type','aspiration',
           'num-of-doors','body-style','drive-wheels','engine-location','wheel-base',
           'length','width','height','curb-weight','engine-type',
           'num-of-cylinders','engine-size','fuel-system','bore','stroke',
           'compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
Data.columns = headers
print(Data)
print(Data.head())
print(Data.describe(include='all'))
print(Data.info)

#Missing values analysis
Data_1 = pd.DataFrame(Data)
Data_1.replace('?', np.NaN, inplace = True)
print(Data_1)

#Checking whether the missing values are present
Missing_value = Data_1.isnull()
print(Missing_value.head())

#Finding the number of missing values
for column in Missing_value.columns.values.tolist():
     print(column)
     print(Missing_value[column].value_counts())
     print("")
    
    
#Dealing with the missing data
#here the missing data is replaced by means, frequency and other functions

#Replace by means    
#normalized-losses": 41 missing data, replace them with mean
#"stroke": 4 missing data, replace them with mean
#"bore": 4 missing data, replace them with mean
#"horsepower": 2 missing data, replace them with mean
#"peak-rpm": 2 missing data, replace them with mean
    
print(Data['normalized-loss'].dtype)    
avg_1 = Data_1['normalized-loss'].astype('float').mean(axis=0) 
print(avg_1)   
Data_1['normalized-loss'].replace(np.nan,avg_1,inplace=True)
print(Data_1['normalized-loss'])


avg_2 = Data_1['stroke'].astype('float').mean(axis=0)
print(avg_2)
Data_1['stroke'].replace(np.nan,avg_2,inplace=True)
print(Data_1['stroke'])
avg_3 = Data_1['bore'].astype('float').mean(axis=0)
print(avg_3)
Data_1['bore'].replace(np.nan,avg_3,inplace=True)
print(Data_1['bore'])
avg_4 = Data_1['horsepower'].astype('float').mean(axis=0)
print(avg_3)
Data_1['horsepower'].replace(np.nan,avg_4,inplace= True)
print(Data_1['horsepower'])
avg_5=Data_1['peak-rpm'].astype('float').mean(axis=0)
print(avg_5)
Data_1['peak-rpm'].replace('np.nan',avg_5, inplace=True)
print(Data['peak-rpm'])

# replacing the nan in number of doors column to the number of repetitions in the coulmn
# for this code such that you print the maximum number of fours appred in the coulmn
# and find the percentage of fours appred in the number of doors column
# using idxmax() to know which values ocunts are appreing the most inthe num-of-doors columns

print(Data_1['num-of-doors'].value_counts())

print(Data_1['num-of-doors'].value_counts().idxmax())
print(Data_1['num-of-doors'].replace(np.nan,'four',inplace=True))
print(Data_1['price'].isna())
print(Data_1['price'].isna().value_counts())
Data_1.dropna(subset=['price'],axis=0,inplace=True)
print(Data_1['price'])
print(Data_1.shape)
# resetting the index as we dropped the rows
Data_1.reset_index(drop=True,inplace=True)
print(Data_1.head())


# Coorecting the data format

print(Data_1.dtypes)
# changing the datatypes of all the attributes given into correct data

Data_1[['normalized-loss']]=Data_1[['normalized-loss']].astype('int')
Data_1[['bore','stroke','horsepower','peak-rpm','price']] = Data_1[['bore','stroke','horsepower','peak-rpm','price']].astype('float')
print("Done")
print(Data_1.dtypes)

#Data Standaradization

#In our dataset, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit. Assume we are developing an application in a country that accept the fuel consumption with L/100km standard
#We will need to apply data transformation to transform mpg into L/100km?
print(Data_1.head())
Data_1['city-L/100km']= 235/Data_1['city-mpg']
print(Data_1.head())
Data_1['highway-L/100km']= 235/Data_1['highway-mpg']
print(Data_1.head())


# Data-normalization
#Why normalization?

#Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variable variance is 1, or scaling variable so the variable values range from 0 to 1
#Example

#To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height"
##Target: we would like to Normalize those variables so their value ranges from 0 to 1.
#Approach: replace origianl value by (original value)/(maximum value)

Data_1['length']=Data_1['length']/Data_1['length'].max()
Data_1['width']=Data_1['width']/Data_1['width'].max()
Data_1['height']=Data_1['height']/Data_1['height'].max()
print('done')


# binning the data
#Why binning?

#Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
##Example:

#In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values. What if we only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? Can we rearrange them into three ‘bins' to simplify analysis?
#We will use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins
print(Data_1.dtypes)
print(min(Data_1['horsepower']))
print(max(Data_1['horsepower']))
binwidth = (max(Data_1['horsepower'])-min(Data_1['horsepower']))/4
print('binwidth',binwidth)
bins = np.arange(min(Data_1['horsepower']),max(Data_1['horsepower']),binwidth)
print(bins)
group_names=['low','medium','high']
Data_1['horsepower-binned']=pd.cut(Data_1['horsepower'],bins,labels=group_names,include_lowest=True)
print(Data_1[['horsepower','horsepower-binned']].head(20))

# bins_viasualization

import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.figure()
plt.pyplot.hist(Data_1['horsepower'],bins,rwidth=0.5)
plt.pyplot.xlabel('horsepower')
plt.pyplot.ylabel('count')
plt.pyplot.title('horse-power for the automobiles')


# dividing the cars with the price range
print(max(Data_1['price']))
print(min(Data_1['price']))
binwidth_1=(max(Data_1['price'])-min(Data_1['price']))/4
print(binwidth_1)
bins_2=np.arange(min(Data_1['price']),max(Data_1['price']),binwidth_1)
print(bins_2)
group_names_2=['low','medium','high']
Data_1['Price-binned']= pd.cut(Data_1['price'],bins_2,labels=group_names_2, include_lowest= True)
print(Data_1[['price','Price-binned']])
 # histogram for price binning
plt.pyplot.figure()
plt.pyplot.hist(Data_1['price'],bins_2,rwidth=0.8)
plt.pyplot.xlabel('price')
plt.pyplot.ylabel('no of cars in range')
plt.pyplot.title('price range in automobiles')



#Indicator variable (or dummy variable)
##What is an indicator variable?

#An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.
#Why we use indicator variables?

#So we can use categorical variables for regression analysis in the later modules.
#Example

#We see the column "fuel-type" has two unique values, "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.
#We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.# creating 
Data_1.columns
dummy_var_1=pd.get_dummies(Data_1['fuel-type'])
print(dummy_var_1.head())
dummy_var_1.rename(columns={'fuel-type-diesel':'gas','fuel-type-diesel':'diesel'},inplace=True)
print(dummy_var_1.head())
Data_1=pd.concat([Data_1,dummy_var_1],axis=1)
print(Data_1)
Data_1.drop(['fuel-type'],axis=1,inplace=True)
print(Data_1.head())
dummy_var_2=pd.get_dummies(Data_1['aspiration'])
print(dummy_var_2)
Data_1=pd.concat([Data_1,dummy_var_2],axis=1)
print(Data_1.head())
Data_1.drop(['aspiration'],axis=1,inplace=True)

# saving the file into a new csv files
Data_1.to_csv('Clean_Auto_Data.csv')

# Main question
# characteristics that have the most impact on  the car price

