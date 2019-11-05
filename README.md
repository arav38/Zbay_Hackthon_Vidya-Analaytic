import pandas as pd
import numpy as np

zbay_traindata = pd.read_csv(r"C:\akshay\hackthon\train.csv",engine = "python")
zbay_testdata = pd.read_csv(r"C:\akshay\hackthon\test.csv",engine = "python")
zbay_viewdata = pd.read_csv(r"C:\akshay\hackthon\view_log.csv",engine = "python")
zbay_traindata.head()
zbay_viewdata.head()

zbay_testdata.shape
zbay_traindata.isnull().sum()
zbay_traindata.dtypes
zbay_traindata_df = pd.DataFrame.copy(zbay_traindata) ##copied the file
zbay_traindata_df = zbay_traindata_df.drop(["user_id"],axis =1)
zbay_traindata_df.isnull().sum()


#creating the list of categorical data
colnames = ["impression_id","impression_time","os_version"]
colnames


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colnames:
    zbay_traindata_df[x]= le.fit_transform(zbay_traindata_df[x])
    
zbay_traindata_df.head()


#0  -----> not click
#1------ click


zbay_testdata.head()

zbay_testdata_df = pd.DataFrame.copy(zbay_testdata)
zbay_testdata_df = zbay_testdata_df.drop(["user_id"],axis =1)
zbay_testdata_df.head()


#creating the list of categorical data
colnames = ["impression_id","impression_time","os_version"]
colnames

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colnames:
    zbay_testdata_df[x]= le.fit_transform(zbay_testdata_df[x])
    
zbay_testdata_df.head()

zbay_testdata_df.dtypes
zbay_traindata_df.shape


# Creating X and Y


x_train = zbay_traindata_df.values[:,:-1]
y_train= zbay_traindata_df.values[:,-1]

y_train = y_train.astype(int)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(x_train)

x_train=scaler.transform(x_train)
print(x_train)
from sklearn.model_selection import train_test_split


#split the data into tesst and train

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.3 )


x_test = zbay_testdata_df.values[:,:]

#creating the model

from sklearn.linear_model import LogisticRegression

#creating model
classifier = LogisticRegression()

#fitting training data to model

classifier.fit(x_train,y_train) #fit function used to train the model

y_pred = classifier.predict(x_val)


print(list(zip(y_pred,x_val)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(y_val,y_pred)
print(cfm)

print()


print("classification report :")

print()

print(classification_report(y_val,y_pred))

print()

acc = accuracy_score(y_val,y_pred)
print("Accuracy of model is",acc)



test_data  = pd.read_csv(r"C:\akshay\hackthon\test.csv",engine = "python" ,header =0)
test_data["Y_predictions"] =  y_pred
test_data.head()

from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(y_val, y_pred[:,-1])
auc = metrics.auc(fpr,tpr)
print(auc)

