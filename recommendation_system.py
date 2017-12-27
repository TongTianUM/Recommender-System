from surprise import SVD, CoClustering, NMF
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.dataset import Reader
from surprise.evaluate import GridSearch
import pandas as pd
from glob import glob
from sklearn.metrics import jaccard_similarity_score as jaccard

# data preparation for collaborative filtering
files=glob('homework/*.txt')
users=pd.read_csv(files[0], sep='\t', header=0)
hotels=pd.read_csv(files[1], sep='\t', header=0)
activity=pd.read_csv(files[2], sep='\t', header=0).drop(16135).reset_index().drop('index', axis=1)  # remove the outlier
activity_count=activity.assign(browse=1).groupby(['user', 'hotel']).count().reset_index()
data=users.assign(key=1).merge(hotels.assign(key=1), on='key', how='inner').drop('key', axis=1)
data=data.merge(activity_count, on=['user', 'hotel'], how='left')
data['browse']=data.browse.fillna(0)
data=data[['user', 'hotel', 'browse']]


# tentatively CV test for some algorithms
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data, reader)

data_cv=data
data_cv.split(n_folds=5)

# SVD test
svd = SVD()
perf = evaluate(svd, data, measures=['RMSE'])
print_perf(perf)      # MSE 0.052

param_svd = {'n_factors': [50, 100], 'lr_all': [0.003, 0.005],
              'reg_all': [0.05, 0.1, 0.5]}
gs = GridSearch(SVD, param_svd, measures=['RMSE'])
gs.evaluate(data_cv) # RMSE 0.2272 ~ 0.2284, after many tests notice 0.2272 is a benchmark, 100, 0.003, 0.1

# Co-clustering test
coc=CoClustering()
perf = evaluate(coc, data, measures=['RMSE'])
print_perf(perf)     # MSE 0.053

param_svd = {'n_cltr_u': [3, 5, 7], 'n_cltr_i': [3, 5, 7],
              'n_epochs': [10, 20]}
gs = GridSearch(CoClustering, param_svd, measures=['RMSE'])
gs.evaluate(data_cv)  # generally worse than SVD here, especially for larger cluster numbers

# Non-negative Matrix Factorization
nmf=NMF()
perf = evaluate(nmf, data, measures=['RMSE'])
print_perf(perf)    # MSE 0.053

param_svd = {'n_factors': [5, 15], 'reg_qi': [0.06, 0.1], 'biased': [True], 'reg_pu': [0.06, 0.1], 'n_epochs': [20, 50]}
gs = GridSearch(NMF, param_svd, measures=['RMSE'])
gs.evaluate(data_cv)  # RMSE 0.2274 ~ 0.33


# do full-data training and testing using SVD
svd=SVD(lr_all=0.003, reg_all=0.1)
trainset=data.build_full_trainset()
svd.train(trainset)
testset=trainset.build_testset()
predictions=svd.test(testset)


def getRecommend(predictions, activity_dic):
    recommend={}
    maxest={}
    for uid, iid, true_r, est, _ in predictions:
        if iid in activity_dic[uid]:
            continue
        if uid not in recommend:
            recommend[uid]=iid
            maxest[uid]=est
        else:
            if est>maxest[uid]:
                recommend[uid]=iid
                maxest[uid]=est
    return recommend


activity_dic={u: list(h) for u,h in activity.groupby('user')['hotel']}
recommend=getRecommend(predictions, activity_dic)
result=pd.DataFrame(recommend.items(), columns=['user', 'hotel'])
result.to_csv('recommendation3.txt', sep='\t', index=False)


