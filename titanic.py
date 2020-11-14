import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Get train data
dataset=pd.read_csv('train.csv')
y_train=dataset.iloc[:,1:2].values
x=dataset.iloc[:,4:12].values
x2=dataset.iloc[:,[2]].values

X_train=np.append(x2,x,1)
X_train=X_train[:,[0,1,2,3,4,5,6,8]]

#Get test data
test_set=pd.read_csv('test.csv')
gender_submision=pd.read_csv('gender_submission.csv')


y_test=gender_submision.iloc[:,1:2].values
test_set.drop(test_set.columns[[0, 2,9]], axis=1, inplace=True)
X_test=test_set.values




#Dealing with missing data in training set
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean')
imputer=imputer.fit(X_train[:,2:3])
X_train[:,2:3]=imputer.transform(X_train[:,2:3])

imputer_2=SimpleImputer(strategy="most_frequent")
imputer_2=imputer_2.fit(X_train[:,7:8])
X_train[:,7:8]=imputer_2.transform(X_train[:,7:8])

#X_train=X_train[:,[0,1,2,3,4,5,6,8]]

#Dealing with missing data in test set
imputer=imputer.fit(X_test[:,2:3])
X_test[:,2:3]=imputer.transform(X_test[:,2:3])

imputer_2=imputer_2.fit(X_test[:,7:8])
X_test[:,7:8]=imputer_2.transform(X_test[:,7:8])
X_test[:,6:7]=imputer.transform(X_test[:,6:7])


#Crating dummie variables in train set
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import  ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [1,5,7])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_train=X_train[:,1:]

#Creating dummie variables in test set
X_test = np.array(ct.transform(X_test))
X_test=X_test[:,1:]


from sklearn.preprocessing import MinMaxScaler
SC=MinMaxScaler(feature_range=(-1,1))
X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)




from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(100,random_state=0,criterion='entropy',)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_train)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)



#Predicting test set
Test_set_predictions=classifier.predict(X_test)
cm2=confusion_matrix(y_test,Test_set_predictions)

#Applying k- fold cross validation
from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,)
acc.mean()
acc.std()


#Applying grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'n_estimators':[80,100,150,180,200], 'criterion':['entropy'],'min_samples_split':[1,2,3,4,5],'max_depth':[1,3,5,8,10]},
            
    ]
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_acuraccy=grid_search.best_score_
best_parameters=grid_search.best_params_



#







