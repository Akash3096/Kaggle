import pandas as pd
import numpy as np

location = 'C:\\Users\\Akash\\.kaggle\\competitions\\titanic\\'

train = pd.read_csv(location + 'train.csv')
test = pd.read_csv(location + 'test.csv')

print("Dimensions of train: {}".format(train.shape))
print("Dimensions of test: {}".format(test.shape))
# print(train.head())
train.drop('PassengerId', axis=1, inplace=True)
# print(train.describe())

# import matplotlib.pyplot as plt
#
# sex_pivot = train.pivot_table(index="Sex",values="Survived")
# sex_pivot.plot.bar()
# plt.show()

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

train["Pclass"].value_counts()

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)

from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
# lr.fit(train[columns], train['Survived'])

holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

# train_X, test_X, train_y, test_y = train_test_split(
#     all_X, all_y, test_size=0.20,random_state=0)
#
# lr = LogisticRegression()
# lr.fit(train_X, train_y)
# predictions = lr.predict(test_X)
#
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(test_y, predictions)
# print(accuracy)


# from sklearn.model_selection import cross_val_score
#
# lr = LogisticRegression()
# scores = cross_val_score(lr, all_X, all_y, cv=10)
# scores.sort()
# accuracy = scores.mean()
#
# print(scores)
# print(accuracy)

lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])

holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)