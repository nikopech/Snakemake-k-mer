# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted',labels=np.unique(y_predicted))
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted',labels=np.unique(y_predicted))
    return accuracy, precision, recall, f1



def fit_func(X,Y):

	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=None)
	# prepare models
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	scoring = 'accuracy'
	result=0
	for name, model in models:
		kfold = model_selection.KFold(n_splits=2, random_state=None)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		result=result+cv_results.mean()
		print(msg)
		
	# boxplot algorithm comparison
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()
	plt.savefig('fffff.png')
	plt.clf()
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=42)
	print(Y_train)
	print(Y_test)
	print(X_test)
	print(X_train)
	classifier = MultinomialNB()
	classifier.fit(X_train, Y_train)
	y_pred = classifier.predict(X_test)
	print("Confusion matrix\n")
	print(pd.crosstab(pd.Series(Y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
	
	accuracy, precision, recall, f1 = get_metrics(Y_test, y_pred)
	print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

	result=result+accuracy
	return result/(len(models)+1)