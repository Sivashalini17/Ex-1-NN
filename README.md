NAME : SIVA SHALINI.S

REG.NO: 212224240154

EX. NO.1

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
df = pd.read_csv("Churn_Modelling.csv")
print(df)
x = df.iloc[:, :-1].values
x
y = df.iloc[:, -1].values
y
print(df.isnull().sum())
df.duplicated()
df.describe()
df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:

DATASET PREVIEW:

<img width="672" height="787" alt="image" src="https://github.com/user-attachments/assets/ab093539-8da0-48c6-a7d8-66725ef06ac0" />

FEATURE MATRIX:

<img width="617" height="142" alt="image" src="https://github.com/user-attachments/assets/1146c8de-4df8-4cde-9310-2caccdf04795" />

TARGET VECTOR:

<img width="265" height="28" alt="image" src="https://github.com/user-attachments/assets/866be389-39bb-4ebd-92e3-d3614ef6d06c" />

CHECK MISSING VALUES:

<img width="197" height="307" alt="image" src="https://github.com/user-attachments/assets/34d7e504-e51c-4a6c-83bb-f80f6c0ccf62" />

CHECK FOR DUPLICATE VALUES:

<img width="188" height="501" alt="image" src="https://github.com/user-attachments/assets/38f46b14-662b-4665-ae30-6ffe564fe6f5" />

DATASET STATISTICAL SUMMARY :

<img width="930" height="206" alt="image" src="https://github.com/user-attachments/assets/672bf95b-b3d3-4ebb-b05b-2928e96ed5a6" />

NORMALIZED DATASET:

<img width="666" height="531" alt="image" src="https://github.com/user-attachments/assets/4a70eb0b-882a-4116-b53b-9939de437eee" />

TRAINING DATA:

<img width="401" height="160" alt="image" src="https://github.com/user-attachments/assets/9c7915b6-30e6-43c5-be03-19eb6e825cb1" />

TESTING DATA:

<img width="412" height="160" alt="image" src="https://github.com/user-attachments/assets/bcd12fdc-52c4-467f-8304-36652d9b1b99" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
