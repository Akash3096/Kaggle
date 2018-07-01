# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import pandas as pd
# import json
#
# # Dataset Preparation
# print ("Read Dataset ... ")
# def read_dataset(path):
# 	return json.load(open(path))
# train = read_dataset('train.json')
# test = read_dataset('test.json')
#
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

train = pd.read_json('train.json')
test = pd.read_json('test.json')
target = train['cuisine']

train['in_text'] = train['ingredients'].apply(lambda x: ' '.join([str(i).lower() for i in x]))
test['in_text'] = test['ingredients'].apply(lambda x: ' '.join([str(i).lower() for i in x]))

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack

def word_count(text, dc):
    text = set(text.split(' '))
    for w in text:
        dc[w] += 1


def remove_high_freq(text, dc):
    return ' '.join([w for w in text.split() if w in dc])


label = LabelEncoder()
y_target = label.fit_transform(target)
print(y_target.shape)
word_count_dict_one = defaultdict(np.uint32)
train['in_text'].apply(lambda x: word_count(x, word_count_dict_one))
sorted_dict = sorted(word_count_dict_one.values(),reverse=True)

freq_words = [key for key in word_count_dict_one if word_count_dict_one[key]>5000]
for key in freq_words:
    word_count_dict_one.pop(key, None)
train['in_text'] = train['in_text'].apply( lambda x : remove_high_freq(x, word_count_dict_one))
tfidf = Tfidf()
train_tf = tfidf.fit_transform(train['in_text'])
test_tf = tfidf.transform(test['in_text'])

def tokenize(text):
    return [w for w in text.split()]


count = CountVectorizer(min_df=20, ngram_range=(1, 2),dtype=np.uint8, tokenizer=tokenize, binary=True )
train_count = count.fit_transform(train['in_text'])
test_count = count.transform(test['in_text'])
final_train = hstack((train_tf, train_count)).tocsr()
final_test = hstack((test_tf, test_count)).tocsr()

print(train_tf.shape)
print(test_tf.shape)
print(train_count.shape)
print(test_count.shape)
print(final_train.shape)
print(final_test.shape)

forest = RandomForestClassifier(n_estimators = 100)
# classifier = GradientBoostingClassifier(loss='deviance', n_estimators=100, learning_rate=0.1)
# classifier.fit(final_train, y_target)
forest.fit(final_train,y_target)
y_pred = forest.predict(final_test)
y_pred = label.inverse_transform(y_pred)
subm = test.copy()
subm['cuisine'] = y_pred
columns = ['ingredients','in_text']
subm.drop(columns, inplace=True, axis=1)
subm.head()
subm.to_csv("first_cook.csv")