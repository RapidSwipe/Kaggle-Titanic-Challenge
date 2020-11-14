# Kaggle-Titanic-Challenge
https://www.kaggle.com/c/titanic

In this challenge I used Random Tree Forest to predict whether the passanger would die or not.
I tuned the parameters using Grid Search
</br>
<code>from sklearn.model_selection import GridSearchCV</code></br>
<code>
parameters=[{'n_estimators':[80,100,150,180,200], 'criterion':['entropy'],'min_samples_split':[1,2,3,4,5],'max_depth':[1,3,5,8,10]},]
</code></br>

<code>grid_search=GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_acuraccy=grid_search.best_score_
best_parameters=grid_search.best_params_</code>

Finally after applying k- fold cross validation</br>
<code>from sklearn.model_selection import cross_val_score</code>
</br>
<code>acc=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,)</code>
</br><code>
acc.mean()</code></br>
<code>
acc.std()
</code>
</br>
Accuracy rose up to </br>
<code>0.8327715355805243</code></br>
And standard deviation equals to</br>
<code>0.0397078821252685</code>
