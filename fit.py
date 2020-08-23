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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
import plot

def fit_func(X,Y,k):

	"""Count accuracy of machine learning algoriths

    Parameters
    ----------
    X : array
        A  2-d array with the entropy and godel numbering
    Y : array
        The value of the class
	k: int
		K-mer value
    Returns
    -------
        An array of the mean accuracy of the comparison algorithms.
	"""
	
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=None)
	# prepare models
	models = []
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('MUNB', MultinomialNB()))
	# evaluate each model in turn
	results = []
	mean_results=[]
	names = []
	scoring = 'accuracy'
	
	for name, model in models:
		kfold = model_selection.KFold(n_splits=2, random_state=None)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		mean_results.append(cv_results.mean())
		
	# boxplot algorithm comparison
	plot.compare_algorithms(results,names,k)
	
	return mean_results
	
	
def pca(X,Y,k):

	"""Count accuracy of PCA method

    Parameters
    ----------
    X : array
        A  2-d array with the entropy and godel numbering
    Y : array
        The value of the class
	k: int
		K-mer value
    Returns
    -------
        The accuracy of the pca method in RandomForestClassifier.
	"""
	# test_size: what proportion of original data is used for test set
	train_img, test_img, y_train, test_y = train_test_split( X, Y, test_size=0.20, random_state=0)
	# Standardizing the features
	scaler = StandardScaler()
	# Fit on training set only.
	scaler.fit(train_img)
	# Apply transform to both the training set and the test set.
	train_img = scaler.transform(train_img)
	test_img = scaler.transform(test_img)
	# Make an instance of the Model
	pca = PCA(n_components=1)
	pca.fit(train_img)
	#Apply the mapping (transform) to both the training set and the test set.
	train_img = pca.transform(train_img)
	test_img = pca.transform(test_img)
	
	X_new = pca.inverse_transform(train_img)
	plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
	plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
	plt.axis('equal');
	
	classifier = RandomForestClassifier(max_depth=2, random_state=0)
	classifier.fit(train_img, y_train)

	# Predicting the Test set results
	y_pred = classifier.predict(test_img)
	
	cm = confusion_matrix(test_y, y_pred)
	print(cm)
	print('Accuracy' + str(accuracy_score(test_y, y_pred)))
	
	return accuracy_score(test_y, y_pred)