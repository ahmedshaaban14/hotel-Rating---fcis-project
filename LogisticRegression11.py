import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Label_Endcoder import *
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

Missing_Val =["NA",' ','']
data = pd.read_csv('hotel-classification-dataset.csv',na_values=Missing_Val)




mean_Additinal_num_Score = data['Additional_Number_of_Scoring'].mean()
mean_Av_Score = data['Average_Score'].mean()
mean_Neg_Count = data['Review_Total_Negative_Word_Counts'].mean()
mean_Numb_Rev = data['Total_Number_of_Reviews'].mean()
mean_Numb_Reviwer_Revs = data['Total_Number_of_Reviews_Reviewer_Has_Given'].mean()
mean_Pos_Count = data['Review_Total_Positive_Word_Counts'].mean()

data['Additional_Number_of_Scoring'].fillna(mean_Additinal_num_Score, inplace=True)
data['Average_Score'].fillna(mean_Av_Score, inplace=True)
data['Review_Total_Negative_Word_Counts'].fillna(mean_Neg_Count, inplace=True)
data['Total_Number_of_Reviews'].fillna(mean_Numb_Rev, inplace=True)
data['Total_Number_of_Reviews_Reviewer_Has_Given'].fillna(mean_Numb_Reviwer_Revs, inplace=True)
data['Review_Total_Positive_Word_Counts'].fillna(mean_Pos_Count, inplace=True)

data['Negative_Review'].fillna('No Negative',inplace=True)
data['Positive_Review'].fillna('No Positive',inplace=True)
data['Hotel_Address'].fillna('No Hotel',inplace=True)
data['Review_Date'].fillna('No Date',inplace=True)
data['Hotel_Name'].fillna('No Name',inplace=True)
data['Reviewer_Nationality'].fillna('No Nationality',inplace=True)
data['Tags'].fillna('No Tags',inplace=True)
data['days_since_review'].fillna('No Days',inplace=True)


cols=('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Reviewer_Score','Tags')

data=Feature_Encoder(data,cols);



needed=['Reviewer_Score','Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Tags','Review_Total_Positive_Word_Counts','Review_Total_Negative_Word_Counts']

def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True)
    return data

data = drop_columns(data, needed)
X = np.array(data.iloc[:, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"][:])
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20 , random_state=32 )

scaler = StandardScaler() #to unit the variance
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier(criterion ='gini', max_depth= 9,min_samples_leaf= 1, min_samples_split= 2)
dt.fit(X_train, y_train)

type=['log',  'knn', 'DecisionTree']
accuracy_list=[]
for i, clf in enumerate((logreg,  knn, dt)): #بلف علي المودلز وببرينت الاكيورسي وال mse وبعمل save للداتا في فايل

    clf.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    accuracy = loaded_model.score(X_test, y_test)
    accuracy_list.append(accuracy)
    prediction = clf.predict(X_test)
    print('\t'+type[i])
    print('accuracy ' + ': ' + str(accuracy * 100) + '%')
    print('Mean Square Error: ', metrics.mean_squared_error(y_test, prediction))
