from glmucb import *
from movie_lens_example import *
import cPickle
import os
import collections
import matplotlib.pyplot as plt


def read_movie_lens_logs(filename):
    log = pd.read_csv(filename,sep='\t',header=None)
    log.columns = ['user_id','item_id','rating','timestamp']
    return log

def getUsers(filename):
    log = read_movie_lens_logs(filename)
    return np.unique(train['user_id'].values)


def generateTrainTest(total_filename,threshold,n):
    
    log = read_movie_lens_logs(total_filename)
    df =log.groupby(['user_id'])['item_id'].count().reset_index()
    user_indices = df[df['item_id']>=threshold]['user_id'].values
    
    #filter test users, and then filter test
    
    user_indices = sorted(np.random.choice(user_indices,n))
    len(np.unique(user_indices))
    test = log[log['user_id'].isin(user_indices)]
    train = log[~log['user_id'].isin(user_indices)]
    train.to_csv('train.csv')
    #train = train.sort(['timestamp'])
    test = test.sort(['timestamp'])
    test.to_csv('test.csv')
    return train,test




rank = 30
#train,test = generateTrainTest('ml-100k/u.data',120,200)
#item_features  = getItemFeatures(train,rank)

train =  pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
item_features = cPickle.load(open('item_features_'+str(rank)+'.p','r'))
#For movie lens
def evaluate(T,test,lambda_,c):
    
    users = np.unique(test.user_id)
    print len(users)
    d =  item_features.shape[0]
    
    algorithm_obj = glmucb([],T,c,d,lambda_)
    
    precisions = [0]*len(users)
    recalls = [0]*len(users)
    df = test.groupby(['user_id'])
    
    
    
    for i,user in enumerate(users):
        #get system rating records:
        records = df.get_group(user)
        user_arms = records['item_id'].values
        max_arm = user_arms[0]
        response = 0
        for t in xrange(T):
            #print t
            #for algorithm,algorithm_obj in algorithms.items():
            if t == 0:
                algorithm_obj.arm_list={i:glmucb_arm(d,c,lambda_) for i in user_arms}
            

            #for arm in algorithm_obj.arm_list:
            algorithm_obj.inference_step(max_arm,item_features,t,response)
            
            max_arm = algorithm_obj.recommend()
            #remove the max arm after recommending it:
            response = records[records['item_id']==max_arm]['rating']
            response = int(response>=4)

            algorithm_obj.updateParams(max_arm,response,item_features)
            precisions[i] += (response/(1.0*len(users)))
                
            recalls[i] += (response/(1.0*len(algorithm_obj.arm_list)*len(users)))


    return precisions,recalls



#getResult('linucb',120)
#c = 0.1, pr,re ->
#c = 1.7, pr,re ->
pr,re = evaluate(10,test,1,2)
print sum(pr)
print sum(re)

pr,re = evaluate(20,test,1,2)
print sum(pr)
print sum(re)

pr,re = evaluate(40,test,1,2)
print sum(pr)
print sum(re)

pr,re = evaluate(120,test,1,2)
print sum(pr)
print sum(re)


