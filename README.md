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

<img width="1563" height="551" alt="image" src="https://github.com/user-attachments/assets/f9a4aac1-9fde-4249-ad56-26c393b82e6c" />

X Values:

<img width="1320" height="290" alt="image" src="https://github.com/user-attachments/assets/ac948d80-ec2f-44b2-8c21-78e6876b1604" />

Y Values:

<img width="810" height="67" alt="image" src="https://github.com/user-attachments/assets/0bc716e8-8387-43e4-943f-17f67d9dbb28" />

CHECK MISSING VALUES:

<img width="475" height="637" alt="image" src="https://github.com/user-attachments/assets/1ad780f8-1d23-404c-b277-fc68d744e0c4" />

CHECK FOR DUPLICATE VALUES:

<img width="533" height="512" alt="image" src="https://github.com/user-attachments/assets/cbe9a776-0100-412d-a681-25f413ad99f5" />

DATASET STATISTICAL SUMMARY :

<img width="1546" height="376" alt="image" src="https://github.com/user-attachments/assets/e748edbe-9641-4c82-8154-dc381ea92219" />

NORMALIZED DATASET:

<img width="1171" height="912" alt="image" src="https://github.com/user-attachments/assets/1392ae2b-c263-4639-93c2-0765c9a684f3" />

TRAINING DATA :

<img width="1070" height="237" alt="image" src="https://github.com/user-attachments/assets/b251bc43-4a3f-4f8b-bd10-a026469a85d2" />

TESTING DATA:

<img width="1101" height="242" alt="image" src="https://github.com/user-attachments/assets/ac23e21a-b7a4-4792-89e1-cdffc7fcc7cc" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
