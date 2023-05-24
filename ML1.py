import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Load the dataset
needed=["Average_Score","Review_Total_Negative_Word_Counts","Review_Total_Positive_Word_Counts",'Reviewer_Score']

def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True) #بغير الداتا ذات نفسخا true
    return data

Missing_Val =["NA",' ','']
df = pd.read_csv('hotel-regression-dataset.csv' , na_values=Missing_Val)


# Data preprocessing
df.fillna(0, inplace=True)
print(df.isnull().values.any())
print(df.isnull().sum())
print(df.describe())
df.drop_duplicates(inplace=True)
df['Review_Date'] = pd.to_datetime(df['Review_Date'])
df['Year'] = df['Review_Date'].dt.year
df['Month'] = df['Review_Date'].dt.month
df['Day'] = df['Review_Date'].dt.day

for i in range(0, len(df.columns)):
    df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')
#mean_lat = df['lat'].mean()
#mean_lng = df['lng'].mean()
mean_Additinal_num_Score = df['Additional_Number_of_Scoring'].mean()
mean_Av_Score = df['Average_Score'].mean()
mean_Neg_Count = df['Review_Total_Negative_Word_Counts'].mean()
mean_Numb_Rev = df['Total_Number_of_Reviews'].mean()
mean_Numb_Reviwer_Revs = df['Total_Number_of_Reviews_Reviewer_Has_Given'].mean()
mean_Pos_Count = df['Review_Total_Positive_Word_Counts'].mean()

#df['lat'].fillna(mean_lat, inplace=True)
#df['lng'].fillna(mean_lng, inplace=True)
df['Additional_Number_of_Scoring'].fillna(mean_Additinal_num_Score, inplace=True)
df['Average_Score'].fillna(mean_Av_Score, inplace=True)
df['Review_Total_Negative_Word_Counts'].fillna(mean_Neg_Count, inplace=True)
df['Total_Number_of_Reviews'].fillna(mean_Numb_Rev, inplace=True)
df['Total_Number_of_Reviews_Reviewer_Has_Given'].fillna(mean_Numb_Reviwer_Revs, inplace=True)
df['Review_Total_Positive_Word_Counts'].fillna(mean_Pos_Count, inplace=True)

df['Negative_Review'].fillna('No Negative',inplace=True)
df['Positive_Review'].fillna('No Positive',inplace=True)
df['Hotel_Address'].fillna('No Hotel',inplace=True)
df['Review_Date'].fillna('No Date',inplace=True)
df['Hotel_Name'].fillna('No Name',inplace=True)
df['Reviewer_Nationality'].fillna('No Nationality',inplace=True)
df['Tags'].fillna('No Tags',inplace=True)
df['days_since_review'].fillna('No Days',inplace=True)

encoder = LabelEncoder()


cols=('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality')
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X



data=Feature_Encoder(df,cols)
data = drop_columns(data, needed)

# Split the data into features and target
X = np.array(data.iloc[:, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"])

def normalize_data_min_max(x_features): #form 0 to 1
    # loop on Each Column (Features in X)
    for i in range(x_features.shape[1]):
        x_features[:, i] = (x_features[:, i] - min(x_features[:, i])) / (max(x_features[:, i]) - min(x_features[:, i]))
    return x_features


X = normalize_data_min_max(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)
prediction = lr.predict(X)

print('Intercept of linear regression model',lr.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))


n_degree = int(input("Please Enter the Degree Of the poly : "))
poly_feature = PolynomialFeatures(degree=n_degree)
x_poly_train = poly_feature.fit_transform(X_train)
x_poly_test = poly_feature.fit_transform(X_test)

cls = LinearRegression()
cls.fit(x_poly_train, y_train)

print("The Mean Square Error for ploy is : ", metrics.mean_squared_error(Y, prediction))
print("The Intercept for poly regression is : ", cls.intercept_)


