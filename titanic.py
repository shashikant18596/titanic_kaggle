#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


# 
# 
# 
# # Generating dataframe of train dataset

# In[3]:


df_train = pd.read_csv("D:\\titanic/train.csv")
df_train.head(2)


# # Generating dataframe of test dataset

# In[4]:


df_test = pd.read_csv("D:\\titanic/test.csv")
df_test.head(2)


# In[5]:


df = pd.concat([df_train,df_test],axis=0)
df


# In[6]:


df.info()


# # dropping those column which is not significant for the model preparation

# In[7]:


df_train.drop(['PassengerId','Cabin'],axis = 1, inplace=True)
df_train.head(10)


# # spliting name by respictive Title

# In[8]:


Name = df_train['Name']
df_train['Title'] = [ i.split('.')[0].split(',')[-1] for i in Name]
df_train.head(10)


# In[9]:


df_train.drop("Name",axis=1,inplace=True)
df_train.head(10)


# # define a class for data cleaning

# In[10]:


def data_cleaning(data):
    d = df_train[data]
    for column in d.columns:
        if df_train[column].isnull().any():
            df_train[column].fillna(df_train[column].median(),inplace = True)


# In[11]:


feature = ['Age','SibSp','Parch','Fare']
data_cleaning(feature)


# In[12]:


df_train['Embarked'].fillna(method='ffill',inplace=True)


# In[13]:


print(df_train.isnull().sum())


# # univariate analysis for feature having numerical datatype

# In[14]:


def plot(feature):
    v = df_train[feature]
    for i in v.columns:
        sns.displot(df_train[i],kde=True)
        plt.show()


# In[15]:


feature = ['Age','Fare']
plot(feature)


# # univariate analysis for feature having categorical datatypes

# In[16]:


def plot(feature):
    v = df_train[feature]
    v_value = v.value_counts()
    plt.figure(figsize = (9,6))
    plt.xticks(rotation = 60)
    plt.bar(v_value.index,v_value)
    plt.show()


# In[17]:


feature = ['Pclass','Sex','SibSp','Parch','Embarked','Title']
for i in feature:
    plot(i)


# # Multivariate Analysis

# In[18]:


sns.pairplot(df_train,hue="Survived")
plt.show()


# In[19]:


df_train.head(10)


# # Finding Outliers

# In[20]:


def detect_outliers(feature):
    threshold = 3
    outliers = []
    data = df_train[feature]
    mean = np.mean(data)
    std = np.std(data)
    for x in data:
        z_score = ( x - mean ) / std
        if z_score > threshold :
            outliers.append(x)
    plt.title(f"{i}")
    sns.boxplot(data=data)
    plt.show()
    print(f"outliers of {feature} column:- \n {outliers}") 


# In[21]:


features = ["Age","SibSp","Parch","Fare"]
for i in features :
    detect_outliers(i)


# In[22]:


df_train.head(10)


# In[23]:


#outliers of Age column:- [71.0, 70.5, 71.0, 80.0, 70.0, 70.0, 74.0]
o = [71.0, 70.5, 71.0, 80.0, 70.0, 70.0, 74.0]
a = df_train['Age']
for i in a:
    if i in o:
        df_train.Age.replace(df_train['Age'].median(),inplace=True)


# In[24]:


# outliers of Parch column:- [5, 5, 3, 4, 4, 3, 4, 4, 5, 5, 6, 3, 3, 3, 5]
o = [5, 5, 3, 4, 4, 3, 4, 4, 5, 5, 6, 3, 3, 3, 5]
a = df_train['Parch']
for i in a:
    if i in o:
        df_train.Parch.replace(df_train['Parch'].median(),inplace=True)


# In[25]:


# outliers of Fare column:- [263.0, 263.0, 247.5208, 512.3292, 247.5208, 262.375, 263.0, 211.5, 227.525, 263.0, 221.7792, 227.525, 512.3292, 211.3375, 227.525, 227.525, 211.3375, 512.3292, 262.375, 211.3375]
o = [263.0, 263.0, 247.5208, 512.3292, 247.5208, 262.375, 263.0, 211.5, 227.525, 263.0, 221.7792, 227.525, 512.3292, 211.3375, 227.525, 227.525, 211.3375, 512.3292, 262.375, 211.3375]
a = df_train['Fare']
for i in a:
    if i in o:
        df_train.Fare.replace(df_train['Fare'].median(),inplace=True)


# In[26]:


# outliers of SibSp column:- [4, 4, 5, 4, 5, 4, 8, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 5, 5, 4, 4, 5, 4, 4, 8, 4, 4, 8, 4, 8]
o = [4, 4, 5, 4, 5, 4, 8, 4, 4, 8, 4, 8, 4, 4, 4, 4, 8, 5, 5, 4, 4, 5, 4, 4, 8, 4, 4, 8, 4, 8]
for i in a:
    if i in o:
        df_train.SibSp.replace(df_train['SibSp'].median(),inplace=True)


# In[27]:


df_train.head(10)


# In[28]:


l = ["SibSp","Parch","Survived","Age","Fare"]
sns.heatmap(df[l].corr(),annot = True,fmt = ".2f")
plt.show()


# In[29]:


df_train.head()


# In[30]:


for i in features :
    detect_outliers(i)


# In[32]:


df_train.head(10)


# In[33]:


df_train=pd.get_dummies(df_train,columns=["Title"])
df_train = pd.get_dummies(df_train, columns=["Embarked"])
df_train = pd.get_dummies(df_train, columns= ["Ticket"], prefix = "T")
df_train["Sex"] = df_train["Sex"].astype("category")
df_train = pd.get_dummies(df_train, columns=["Sex"])
df_train.head(2)


# In[34]:


df_train.head(10)


# In[50]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[51]:


x = df_train.drop("Survived", axis = 1)
y = df_train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)


# In[52]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[53]:


y_predicted=model.predict(x_test)


# In[54]:


model.score(x_test,y_test)


# In[55]:


from sklearn.metrics import r2_score,mean_squared_error
print(f'R^2 : {r2_score(y_test,y_predicted)}')
print(f'MSE : {mean_squared_error(y_test,y_predicted)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predicted))}')


# In[56]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_grid_parameter = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_grid_parameter = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_grid_parameter = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

lr_grid_parameter = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_grid_parameter = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_grid_parameter,
                   svc_grid_parameter,
                   rf_grid_parameter,
                   lr_grid_parameter,
                   knn_grid_parameter]


# In[58]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[59]:


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# In[63]:


poll = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
poll = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))


# In[ ]:




