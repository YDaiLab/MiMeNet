# Third-party libraries
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def train(train, test, seed=42, feature_select=True):

	x, y = train
	test_x, test_y = test
	num_trees = 500
	clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
	clf.fit(x, y)
	
	feature_importance = clf.feature_importances_
	
	feature_ranking = np.flip(np.argsort(feature_importance))
	num_features = x.shape[1]
	best_num_features = num_features

	if feature_select:
		percent_features = [1.0, 0.75, 0.5, 0.25]

		skf = KFold(n_splits=5, shuffle=True)
	
		best_score = -1
	
		for percent in percent_features:
			run_score = -1
			run_probs = []
			i=0            
			for train_index, valid_index in skf.split(x):
				train_x, valid_x = x[train_index], x[valid_index]
				train_y, valid_y = np.argmax(y[train_index], axis=1), np.argmax(y[valid_index], axis=1)
				
				features_using = int(round(num_features * percent))
				feature_list = feature_ranking[0:features_using]
				filtered_train_x = train_x[:,feature_list]
				filtered_valid_x = valid_x[:,feature_list]
				clf = RandomForestClassifier(n_estimators=200, n_jobs=-1).fit(filtered_train_x, train_y)
				probs = np.array(clf.predict_proba(filtered_valid_x))
				if i == 0:
					run_probs = np.array(probs)
				else:
					run_probs = np.concatenate((run_probs, probs), axis=0)  
				i+=1                
			run_score = roc_auc_score(y, run_probs, average="weighted")
		
			if run_score > best_score:
				best_num_features = num_features

			
	feature_list = feature_ranking[0:best_num_features]
	x_filt = x[:,feature_list]
	test_x_filt = test_x[:,feature_list]
	
	
	clf = RandomForestClassifier(n_estimators=200, n_jobs=-1).fit(x_filt, np.argmax(y, axis=1))

	test_probs = np.array(clf.predict_proba(test_x_filt))


	return  test_probs
