
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('telecomchurn.csv')
# Remove rows with entry errors 
dataset.drop(dataset.index[[5218,3331,4380,753,3826,1082,488,1340,6754,6670,936]],inplace=True)
dataset.TotalCharges = pd.DataFrame(dataset.TotalCharges, dtype='float')
dataset.SeniorCitizen = pd.DataFrame(dataset.SeniorCitizen, dtype='object')

X = dataset.drop(columns=['customerID','Churn'])
y = dataset.Churn
y = y.map(dict(Yes=1, No=0)).values


#Investigate data

dataset.isnull().sum() #checking for total null values

f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['Churn'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Churn')
ax[0].set_ylabel('')
sns.countplot('Churn',data=dataset,ax=ax[1])
ax[1].set_title('Churn')
plt.show()


f,ax=plt.subplots(1,3,figsize=(18,8))
sns.distplot(dataset.TotalCharges,ax=ax[0])
ax[0].set_title('Total Amount Charged')
sns.distplot(dataset.MonthlyCharges,ax=ax[1])
ax[1].set_title('Monthly Charged')
sns.distplot(dataset.MonthlyCharges,ax=ax[2])
plt.show()


f,ax=plt.subplots(1,3,figsize=(18,8))
sns.distplot(dataset.tenure,ax=ax[0])
ax[0].set_title('Tenure')
sns.distplot(dataset.MonthlyCharges,ax=ax[1])
ax[1].set_title('Monthly Charges')
sns.distplot(dataset.TotalCharges,ax=ax[2])
ax[2].set_title('Total Charges')
plt.show()


f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['Contract'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number Of Customers By Contract')
ax[0].set_ylabel('Count')
sns.countplot('Contract',hue='Churn',data=dataset,ax=ax[1])
ax[1].set_title('Contract:Leave vs Stay')
plt.show()


f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['PaymentMethod'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number Of Customers By Payment Method')
ax[0].set_ylabel('Count')
sns.countplot('PaymentMethod',hue='Churn',data=dataset,ax=ax[1])
ax[1].set_title('Payment Method:Leave vs Stay')
plt.show()

f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['InternetService'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number Of Customers By Internet Service')
ax[0].set_ylabel('Count')
sns.countplot('InternetService',hue='Churn',data=dataset,ax=ax[1])
ax[1].set_title('Internet Service:Leave vs Stay')
plt.show()


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("PaymentMethod","tenure", hue="Churn", data=dataset,split=True,ax=ax[0])
ax[0].set_title('PaymentMethod and tenure vs Churn')
ax[0].set_yticks(range(0,100,10))
sns.violinplot("PaymentMethod","tenure", hue="Churn", data=dataset,split=True,ax=ax[1])
ax[1].set_title('PaymentMethod and Tenure vs Churn')
ax[1].set_yticks(range(0,100,10))
plt.show()



f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('MultipleLines',data=dataset,ax=ax[0,0])
ax[0,0].set_title('No. Of Customers MultipleLines')
sns.countplot('MultipleLines',hue='gender',data=dataset,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for MultipleLines')
sns.countplot('MultipleLines',hue='Churn',data=dataset,ax=ax[1,0])
ax[1,0].set_title('MultipleLines vs Churn')
sns.countplot('MultipleLines',hue='PaymentMethod',data=dataset,ax=ax[1,1])
ax[1,1].set_title('MultipleLines vs PaymentMethod')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(dataset[dataset['Contract']=='Month-to-month'].MonthlyCharges,ax=ax[0])
ax[0].set_title('Charges by Month-to-month')
sns.distplot(dataset[dataset['Contract']=='One year'].MonthlyCharges,ax=ax[1])
ax[1].set_title('Charges by One year')
sns.distplot(dataset[dataset['Contract']=='Two year'].MonthlyCharges,ax=ax[2])
ax[2].set_title('Charges by Two year')
plt.show()

f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(dataset[dataset['Contract']=='Month-to-month'].TotalCharges,ax=ax[0])
ax[0].set_title('Charges by Month-to-month')
sns.distplot(dataset[dataset['Contract']=='One year'].TotalCharges,ax=ax[1])
ax[1].set_title('Charges by One year')
sns.distplot(dataset[dataset['Contract']=='Two year'].TotalCharges,ax=ax[2])
ax[2].set_title('Charges by Two year')
plt.show()

sns.heatmap(dataset.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
plt.show()



# Check the skew of all numerical features

from scipy import stats
from scipy.stats import norm, skew 

numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness)

fig = plt.figure()
res = stats.probplot(dataset['TotalCharges'], plot=plt)
plt.show()

#Transform TotalCharges
dataset['TotalCharges'] = stats.boxcox(dataset['TotalCharges'])[0]

sns.distplot(dataset['TotalCharges'] , fit=norm);

fig = plt.figure()
res = stats.probplot(dataset['TotalCharges'], plot=plt)
plt.show()

# Encoding categorical data
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_1 = X.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
X_1 = X_1.apply(le.fit_transform)
X_3 = X.select_dtypes(include=[float,int])

X = X_1.join(X_3).values

onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
#Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0,0]+cm[1,1])/(len(y_pred)))

# Importing the Keras libraries and packages

# Initialising the ANN
classifier2 = Sequential()
# Adding the input layer and the first hidden layer
classifier2.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
#Adding the second hidden layer
classifier2.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier2.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier2.fit(X_train, y_train, batch_size = 32, epochs = 200)
# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)
y_pred2 = (y_pred2 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print((cm2[0,0]+cm2[1,1])/(len(y_pred2)))

# Importing the Keras libraries and packages

# Initialising the ANN and add dropout
classifier3 = Sequential()
classifier3.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
classifier3.add(Dropout(p = 0.2))
classifier3.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier3.add(Dropout(p = 0.2))
classifier3.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier3.fit(X_train, y_train, batch_size = 32, epochs = 200)
# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)
y_pred3 = (y_pred3 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
print((cm3[0,0]+cm3[1,1])/(len(y_pred)))

# Importing the Keras libraries and packages

# Initialising the ANN
classifier4 = Sequential()
# Adding the input layer and the first hidden layer
classifier4.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
classifier4.add(Dropout(p = 0.5))
classifier4.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier4.add(Dropout(p = 0.5))
# Adding the output layer
classifier4.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier4.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier4.fit(X_train, y_train, batch_size = 32, epochs = 500)
# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred4 = classifier4.predict(X_test)
y_pred4 = (y_pred4 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)
print((cm4[0,0]+cm4[1,1])/(len(y_pred4)))

# Importing the Keras libraries and packages

# Initialising the ANN
classifier5 = Sequential()
# Adding the input layer and the first hidden layer
classifier5.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
classifier5.add(Dropout(p = 0.5))
classifier5.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))
classifier5.add(Dropout(p = 0.5))
classifier5.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))
classifier5.add(Dropout(p = 0.5))
classifier5.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))
classifier5.add(Dropout(p = 0.5))
# Adding the output layer
classifier5.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier5.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier5.fit(X_train, y_train, batch_size = 32, epochs = 500)
# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred5 = classifier5.predict(X_test)
y_pred5 = (y_pred5 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred5)
print((cm5[0,0]+cm5[1,1])/(len(y_pred)))

# Importing the Keras libraries and packages

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100,200,300,500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.2))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100,200,300,500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 117))
    classifier.add(Dropout(p = 0.5))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100,200,300,500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Make Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(X_train,y_train)
prediction3=model.predict(X_test)
metrics.accuracy_score(prediction3,y_test)
lcm = confusion_matrix(prediction3,y_test)