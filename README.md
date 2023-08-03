# Swiggy_Dataset_Anlayis
# Import All Necessary Libraries of Python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import Datasets
df=pd.read_csv('swiggy.csv')
df
df.shape

# Return top 5 rows in the Datasets
df.head()

# Return bottom 5 rows in the Datasets
df.tail()

# Data Cleaning

# Check for Null values in the Datasets
df.isnull().sum()

# Check the Datatypes of each column in the Datasets
df.dtypes

for i in df.columns:
    print(i,df[i].sort_values().unique(),'\n', sep='\n')
    
# Exploratory Data Analysis(EDA Analysis)

#To check the co-relation of different variables to each other by using heatmap.
sns.heatmap(df.corr(), annot=True)

# Calculate the average price of food item for each of the different cities.
Average_Price_for_Each_City=df.groupby('City')['Price'].mean()
Average_Price_for_Each_City

# Caculate the top 10 highest average rating getting by each restaurant.
Highest_Rating_by_Each_Restaurant=df.groupby('Restaurant')['Avg ratings'].mean().nlargest(10)
Highest_Rating_by_Each_Restaurant

# Filter restaurants with a avg rating greater than 4.5 and delivery time less than 30 minutes.
High_Rated_Restaurants=df[(df['Avg ratings'] > 4.5) & (df['Delivery time'] < 30)]
High_Rated_Restaurants

# Filter which particular area of Bengaluru serves the food items  at lower cost when price less than 100.
Lower_Cost_in_terms_of_food=df[(df['City'] == 'Bangalore') & (df['Price']<100)]
Lower_Cost_in_terms_of_food

# Filter the Average delivery time for each restaurant for city?
Average_Delivery_Time=df.groupby(['Restaurant', 'City'])['Delivery time'].mean()
Average_Delivery_Time

# Filter the average price of the each food cusines?
Average_Price_of_food=df.groupby('Food type')['Price'].mean()
Average_Price_of_food

# Filter what are most popular cuisines among customers?
popular_cuisines=df['Food type'].value_counts().head()
popular_cuisines

# Create a Scatter plot for Price vs Total ratings
plt.figure(figsize=(8, 6))
plt.scatter(df['Price'], df['Total ratings'], s=df['Avg ratings']*50, alpha=0.5)
plt.xlabel('Price')
plt.ylabel('Total Ratings')
plt.title('Price vs Total Ratings')
plt.grid(True)
plt.show()

# Create a Top 10 areas with the Highest Delivery time
Top_10_Areas=df.groupby('Area')['Delivery time'].sum().nlargest(10)
print(Top_10_Areas)
plt.figure(figsize=(10,6))
sns.barplot(x=Top_10_Areas.index,y=Top_10_Areas.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Restaurant Name')
plt.ylabel('Total Dining Votes')
plt.title('Top 10 Areas with Highest Delivery Time')
plt.show()

# Distribution of Avg Ratings and Total Ratings
plt.figure(figsize=(10,6))
sns.histplot(df['Avg ratings'], kde=True, color='Blue')
sns.histplot(df['Total ratings'], kde=True, color='Red')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Average and Total Ratings')
plt.show()

df.hist(figsize=(14,14),color='Blue')

# Distribution of Prices for food items
plt.figure(figsize=(10,6))
sns.histplot(df['Price'], kde=True, color='Red')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices for food items')
plt.show()

# Average Prices for in Different Cities
plt.figure(figsize=(10,6))
sns.barplot(x='City', y='Price',data=df)
plt.xlabel('Price')
plt.ylabel('City')
plt.title('Distribution of Average Prices in Different Cities')
plt.show()

# Count the occurrences of each food type among all restaurants
food_type_counts=df['Food type'].str.split(',', expand=True).stack().value_counts()

# Select the top 5 food types
top_5_food_types=food_type_counts.head(5)

# Pie chart for top 5 Restaurants by food types
plt.figure(figsize=(8,6))
plt.pie(top_5_food_types, labels=top_5_food_types.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Top 5 Food Types among Restaurants')
plt.show()

# Separete the Categorical Column and Numberical Column
cat=[]
num=[]
for i in df.columns:
    if df[i].dtypes=="O":
        cat.append(i)
    else:
        num.append(i)
cat
num
num.remove('Price')

# Stastical Summary of Datasets
df.describe()

# Outliers Handling
for i in num:
    plt.figure()
    sns.boxplot(y=i, data=df)
    
for i in num:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    iqr=q3-q1
    ll=q1-1.5*iqr
    ul=q3+1.5*iqr
    df=df[(df[i] > ll) & (df[i] < ul)]
    plt.figure()
    sns.boxplot(y=i, data=df)
    
df.shape

# Data Preprocessing

# Scaling the Datasets
sc=StandardScaler()
df[num]=sc.fit_transform(df[num])
df

# Encode the Datasets
df_new=pd.get_dummies(df, columns=cat, drop_first=True)
df_new

# Prepare The Data for Modeling
x=df_new.drop('Price', axis=1)
y=df_new['Price']

# Split the Data into training and testing sets(75% train, 25% test)
X_train, X_test, Y_train, Y_test=train_test_split(x, y, test_size=0.25)

# Create LinearRegression Model
model=LinearRegression()

# Train the LinearRegression Model
model.fit(X_train, Y_train)

# Predict the test set
Y_pred=model.predict(X_test)

# Effect on  Model Evaluation matrics
mse=metrics.mean_squared_error(Y_test, Y_pred)
rmse=np.sqrt(mse)
r2_score=metrics.r2_score(Y_test, Y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-Squared:', r2_score)
