# # Text Data Features
# print ("Prepare text data of Train and Test ... ")
# def generate_text(data):
# 	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
# 	return text_data
#
# train_text = generate_text(train)
# test_text = generate_text(test)
# target = [doc['cuisine'] for doc in train]
#
# # Feature Engineering
# print ("TF-IDF on text data ... ")
# tfidf = TfidfVectorizer(binary=True)
# def tfidf_features(txt, flag):
#     if flag == "train":
#     	x = tfidf.fit_transform(txt)
#     else:
# 	    x = tfidf.transform(txt)
#     x = x.astype('float16')
#     return x
#
# X = tfidf_features(train_text, flag="train")
# X_test = tfidf_features(test_text, flag="test")
#
# # Label Encoding - Target
# print ("Label Encode the Target Variable ... ")
# lb = LabelEncoder()
# y = lb.fit_transform(target)
#
# # Model Training
# print ("Train the model ... ")
# classifier = SVC(C=100, # penalty parameter, setting it to a larger value
# 	 			 kernel='rbf', # kernel type, rbf working fine here
# 	 			 degree=3, # default value, not tuned yet
# 	 			 gamma=1, # kernel coefficient, not tuned yet
# 	 			 coef0=1, # change to 1 from default value of 0.0
# 	 			 shrinking=True, # using shrinking heuristics
# 	 			 tol=0.001, # stopping criterion tolerance
# 	      		 probability=False, # no need to enable probability estimates
# 	      		 cache_size=200, # 200 MB cache size
# 	      		 class_weight=None, # all classes are treated equally
# 	      		 verbose=False, # print the logs
# 	      		 max_iter=-1, # no limit, let it run
#           		 decision_function_shape=None, # will use one vs rest explicitly
#           		 random_state=None)
# model = OneVsRestClassifier(classifier, n_jobs=4)
# model.fit(X, y)
#
# # Predictions
# print ("Predict on test data ... ")
# y_test = model.predict(X_test)
# y_pred = lb.inverse_transform(y_test)
#
# # Submission
# print ("Generate Submission File ... ")
# test_id = [doc['id'] for doc in test]
# sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
# sub.to_csv('svm_output.csv', index=False)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

train = pd.read_json('data/all/train.json')
test = pd.read_json('data/all/test.json')
target = train['cuisine']

train['in_text'] = train['ingredients'].apply(lambda x: ','.join([str(i).lower() for i in x])).astype(str)
test['in_text'] = test['ingredients'].apply(lambda x: ','.join([str(i).lower() for i in x])).astype(str)

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack

label = LabelEncoder()
y_target = label.fit_transform(target)
print(y_target.shape)

tfidf = Tfidf()
train_tf = tfidf.fit_transform(train['in_text'])
test_tf = tfidf.transform(test['in_text'])

def tokenize(text):
    return [w for w in text.split()]


count = CountVectorizer(min_df=20, ngram_range=(1, 3),dtype=np.uint8, tokenizer=tokenize, binary=True )
train_count = count.fit_transform(train['in_text'])
test_count = count.transform(test['in_text'])

final_train = hstack((train_tf, train_count)).tocsr()
final_test = hstack((test_tf, test_count)).tocsr()

# print(train_tf.shape)
# print(test_tf.shape)
# print(train_count.shape)
# print(test_count.shape)
print(final_train.shape)
print(final_test.shape)

import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(final_train, y_target, test_size=0.33, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
# grad.fit(X_train,y_train)
# y_pred = grad.predict(X_test)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
ypred = [int(i) for i in y_pred]
print(accuracy_score(ypred, y_test))
# y_pred = label.inverse_transform(y_pred)



# classifier = GradientBoostingClassifier(loss='deviance', n_estimators=100, learning_rate=0.1)
# classifier.fit(final_train, y_target)



# forest.fit(final_train,y_target)
# y_pred = forest.predict(final_test)
# y_pred = label.inverse_transform(y_pred)
#
#
# subm = pd.read_csv('data/all/sample_submission.csv')
# subm['cuisine'] = y_pred
# columns = ['ingredients','in_text']
#
# subm.to_csv("first_cook.csv", index=False)