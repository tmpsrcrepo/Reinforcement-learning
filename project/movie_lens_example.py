import pandas as pd
import numpy as np
import nimfa

def loadData(dir_):
    user_item_log = pd.read_table(dir_+'u.data',sep='\t',header=None)
    user_item_log.columns = ['user_id','item_id','rating','timestamp']


    item = pd.read_table(dir_+'u.item',sep='|',header=None)
    
    item.columns = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Childrens | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'.split(' | ')


    user = pd.read_table(dir_+'u.user',sep='|',header=None)
    user.columns='user id | age | gender | occupation | zip code'.split(' | ')

    train = pd.read_table(dir_+'ua.base',sep='|',header=None)
    train.columns= ['user_id','item_id','rating','timestamp']
    test = pd.read_table(dir_+'ua.test',sep='\t',header=None)
    test.columns=['user_id','item_id','rating','timestamp']

def read(fname):
    V = np.ones((943, 1682)) * 2.5
    for line in open(fname+'.base'):
        user,item,rating,time = list(map(int,line.split()))
        V[user-1,item-1] = rating
    return V


def readDF(df):
    num_users = len(np.unique(df.user_id.values))
    #num_items = len(np.unique(df.item_id.values))
    val = max(df.rating.values)/2.0
    V = np.ones((943,1682)) * val
    for row in df.as_matrix():
        user,item,rating,time = row
        V[user-1,item-1] = rating
    return V


def factorize(V,rank_,algorithm='snmf'):
    if algorithm == 'snmf':
        snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank_, max_iter=30, version='r', eta=1.,beta=1e-4, i_conv=10, w_min_change=0)
        print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
        fit = snmf()
        sparse_w, sparse_h = fit.fit.sparseness()
        print("""Stats:- iterations: %d - Euclidean distance: %5.3f - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,fit.distance(metric='euclidean'),sparse_w, sparse_h))
        return fit.basis(), fit.coef()
    return 1,1

def run(dir_,rank_):
    for data_set in ['ua','ub']:
        V = read(dir_+data_set)
        W,H = factorize(V,rank_,algorithm='snmf')
        print evaluate(W,H,dir_+data_set)

def evaluate(W,H,fname):
    num_correct = 0
    num_predicted_correct = 0
    for line in open(fname+'.test'):
        user,item,rating,time = list(map(int,line.split()))
        sc=max(min((W[user - 1,:]*H[:,item-1])[0,0],5),1)
        
        if sc >= 4:
            #print sc
            num_predicted_correct+=1
            if rating>=4:
                num_correct+=1
    return num_correct/943.0

#run('ml-100k/',30)








