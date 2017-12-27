from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
import scipy as sc
from sklearn.metrics import mean_squared_error, f1_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.ensemble import RakelD

files=glob('homework/*.txt')

# import the three related data sets
users=pd.read_csv(files[0], sep='\t', header=0)
hotels=pd.read_csv(files[1], sep='\t', header=0)
activity=pd.read_csv(files[2], sep='\t', header=0)  # no missing values

# merge the three data sets into one
activity_count=activity.assign(visit=1).groupby(['user', 'hotel']).count().reset_index()# user 4232 visits a hotel twice
data=users.assign(key=1).merge(hotels.assign(key=1), on='key', how='inner').drop('key', axis=1)
data=data.merge(activity_count, on=['user', 'hotel'], how='left')
data['visit']=data.visit.fillna(0)

y=data.pivot(index='user', columns='hotel', values='visit') # to wide form and retain the labels
temp=data.drop(['hotel', 'visit'], axis=1).loc[data.visit!=1.0]
X=users
X[['mean_rating', 'var_rating']]=temp[['user', 'star_rating']].groupby('user').agg(['mean', np.var]).reset_index().drop('user', axis=1)
X.gender=pd.get_dummies(X.gender).female    # set dummy, female as 1
X=X.fillna(0)  # fill the nans in 'var' that is caused by single observation

t=y
t.iloc[4231, 1]=1.0   # for multi-label classification, manually modify the outlier
X_train, X_test, t_train, t_test= train_test_split(X, t, test_size=0.25, random_state=666) # train-test split

# to sparse matrix for speeding up, and remove user to avoid over-fitting
X_train=sc.sparse.csr_matrix(X_train.drop('user', axis=1).values)
t_train=sc.sparse.csr_matrix(t_train.values)
X_test=sc.sparse.csr_matrix(X_test.drop('user', axis=1).values)
t_test=sc.sparse.csr_matrix(t_test.values)

X_train_scale=scale(X_train.toarray())   # scaling not work well for many methods, for its offset of similarities
X_test_scale=scale(X_test.toarray())

X_sparse=sc.sparse.csr_matrix(X.drop('user', axis=1).values)
t_sparse=sc.sparse.csr_matrix(t.values)

# firstly test the transformations with a simple naive-bayes classifier, roughly conclude that BR suits the most
# intuitively the hotels shouldn't have correlation based on userID, for its randomness
classifier = BinaryRelevance(GaussianNB())
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0
mean_squared_error(t_test.toarray(), probabilities.toarray())   # 0.063299324514418692

classifier = ClassifierChain(GaussianNB())
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test,predictions)    # 0
mean_squared_error(t_test.toarray(), probabilities.toarray())   # 0.084135897849476421

classifier = LabelPowerset(GaussianNB())
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test,predictions)     # 0
mean_squared_error(t_test.toarray(), probabilities.toarray())   # 0.06929592253285459


classifier = MLkNN(k=11)
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test,predictions)     # 0
mean_squared_error(t_test.toarray(), probabilities.toarray())    # 0.055573140385775308

# test SVM with BR
classifier = BinaryRelevance(SVC(probability=True, kernel='rbf'))
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0
mean_squared_error(t_test.toarray(), probabilities.toarray())  # 0.049318199562258865

# test Boosting with BR
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
classifier = BinaryRelevance(ada)
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0.0035211267605633804
mean_squared_error(t_test.toarray(), probabilities.toarray())  # 0.2037614828233037

# test Random Forest with BR
classifier = BinaryRelevance(RandomForestClassifier())
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0.0035211267605633804
mean_squared_error(t_test.toarray(), probabilities.toarray())  # 0.052738170188038475

# test multi-ensemble choices, Space partitioning with a clusterer
base_classifier = RandomForestClassifier()
problem_transform_classifier = LabelPowerset(classifier=base_classifier)
clusterer = IGraphLabelCooccurenceClusterer('infomap', weighted=True, include_self_edges=True)
classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer) # setup the ensemble metaclassifier
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)  # all zero using SVC
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0.029049295774647887, the result is reasonable
mean_squared_error(t_test.toarray(), probabilities.toarray())
# by trying different cluster methods, the walktrap initially is 0.043, greedy 0.029, infomap 0.051
# with a naive test of boosting with some combinations of parameters, forest is better than boosting

# Rakel
base_classifier = RandomForestClassifier()
problem_transform_classifier = LabelPowerset(classifier=base_classifier)
classifier = RakelD(problem_transform_classifier, labelset_size=3) # setup the ensemble meta-classifier
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
accuracy_score(t_test, predictions)   # 0.0079225352112676055, random partition is not good here
mean_squared_error(t_test.toarray(), probabilities.toarray())

# parameter tuning of space partitioning with clusterer
parameters = {
    'classifier': [LabelPowerset()],    # BinaryRelevance performs pretty bad here
    'clusterer': [IGraphLabelCooccurenceClusterer('infomap', weighted=True, include_self_edges=True)],
    #'clusterer__method': ['walktrap', 'infomap'],
    'classifier__classifier': [RandomForestClassifier(n_jobs=-1)],
    'classifier__classifier__n_estimators': [50, 100, 300],    # 100
    'classifier__classifier__min_samples_leaf': [1, 3, 5, 10]   # 3
}
gs = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring='f1_weighted', cv=5) # imbalanced scoring
gs.fit(X_train, t_train)
predictions = gs.predict(X_test)
accuracy_score(t_test, predictions)   # infomap 0.059
f1_score(t_test, predictions, average='weighted')   # 0.247
recall_score(t_test, predictions, average='weighted')  # 0.254, 25% can be predicted, and further over-fitting happens
vote1=pd.DataFrame(gs.predict(X_sparse).toarray())  # construct the binary vote

# transformation methods with SVC grid-search
parameters = {
    'classifier': [SVC(probability=True)],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': np.logspace(-5, 3, 6),
    'classifier__C': np.logspace(-5, 3, 6)
}
gs = GridSearchCV(BinaryRelevance(), parameters, scoring='f1_weighted', cv=4)
gs.fit(X_train, t_train)
probabilities=gs.predict_proba(X_test)
mean_squared_error(t_test.toarray(), probabilities.toarray())

# due to computation limit, try empirical method for SVC instead
classifier = BinaryRelevance(SVC(probability=True, kernel='rbf', gamma=0.001, C=1000000))
classifier.fit(X_train, t_train)
predictions = classifier.predict(X_test)    # 0.01, 100000
probabilities = classifier.predict_proba(X_test)
mean_squared_error(t_test.toarray(), probabilities.toarray())  # 0.0484
recall_score(t_test, predictions, average='weighted')  # 0.065
f1_score(t_test, predictions, average='weighted')  # 0.1
vote2=pd.DataFrame(classifier.predict_proba(X_sparse).toarray())

# fitting full data
base_classifier = RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_leaf=3)
problem_transform_classifier = LabelPowerset(classifier=base_classifier)
clusterer = IGraphLabelCooccurenceClusterer('infomap', weighted=True, include_self_edges=True)
classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)
classifier.fit(X_sparse, t_sparse)
vote1=pd.DataFrame(classifier.predict(X_sparse).toarray())
recall_score(pd.DataFrame(t_sparse.toarray()), vote1, average='weighted')  # 0.63
accuracy_score(pd.DataFrame(t_sparse.toarray()), vote1)  # 0.37

classifier = BinaryRelevance(SVC(probability=True, kernel='rbf', gamma=0.001, C=1000000))
classifier.fit(X_sparse, t_sparse)
predictions = classifier.predict(X_sparse)
probabilities = classifier.predict_proba(X_sparse)
vote2=pd.DataFrame(classifier.predict_proba(X_sparse).toarray())
mean_squared_error(pd.DataFrame(t_sparse.toarray()), vote2)  # 0.049
recall_score(pd.DataFrame(t_sparse.toarray()), predictions, average='weighted')  # 0.079

# construct the numeric vote
score=vote1+vote2


def getRecommend(score, activity):
    n, m=score.shape[0], score.shape[1]
    recommend={'user':[], 'hotel':[]}
    for i in range(n):
        temp=score.iloc[i,:].drop(activity.hotel.loc[activity.user==i+1]-1)
        recommend['user'].append(i+1)
        recommend['hotel'].append(temp.idxmax()+1)
    return recommend


recommend=getRecommend(score, activity)
result=pd.DataFrame(recommend)
result[['user', 'hotel']].to_csv('recommendation4.txt', sep='\t', index=False)

aa=pd.read_csv('recommendation3.txt', sep='\t', header=0)
print(sum(aa.hotel==result.hotel))
# 1486 in common as the recommendation system
