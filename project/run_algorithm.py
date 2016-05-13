from linucb import *
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
    
    user_indices = sorted(np.random.choice(user_indices,n,replace=False))
    len(np.unique(user_indices))
    test = log[log['user_id'].isin(user_indices)]
    train = log[~log['user_id'].isin(user_indices)]
    train.to_csv('train.csv')
    #train = train.sort(['timestamp'])
    test = test.sort(['timestamp'])
    test.to_csv('test.csv')
    return train,test


#featurevec using pmf
def getItemFeatures_response(train,rank,algorithm = 'snmf'):
    Ratings =readDF(train)
    User_Features, Item_Features = factorize(Ratings,rank)
    return Item_Features




#train set: generate item feature vector by PMF
#test set: run LinUCB, glmUCB, epsilon greedy in T=40,60,120 and compare their regrets, cumulative precision, cumulative recall


def getItemFeatures(train,rank):
    #if not os.path.isfile('item_features_'+str(rank)+'.p'):
    #based on the training set -> features
    item_features = getItemFeatures_response(train,rank)
    #export
    cPickle.dump(item_features,open('item_features_'+str(rank)+'.p','w'))
    return item_features
    #else:
    #return cPickle.load(open('item_features_'+str(rank)+'.p','r'))


rank = 30
#train,test = generateTrainTest('ml-100k/u.data',120,200)
#

train =  pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
item_features  = getItemFeatures(train,rank)
#item_features = cPickle.load(open('item_features_'+str(rank)+'.p','r'))

#For movie lens
def evaluateAlgorithms_MovieLens(T,test,item_features,lambda_,alpha,c):
    
    users = np.unique(test.user_id)
    print len(users)
    d = item_features.shape[0]
    #print item_features.shape
    #print item_features[:,0]
    algorithms = {'linucb':linucb([],d,lambda_,alpha)}
    #algorithms = {'glmucb':linucb([],d,lambda_,alpha)}
    
    #user_arms = collections.defaultdict(list) #user: [arms]
    precisions = {algorithm:[0]*len(users) for algorithm in algorithms}
    recalls = {algorithm:[0]*len(users) for algorithm in algorithms}
    df = test.groupby(['user_id'])
    
    for i,user in enumerate(users):
        #get system rating records:
        records = df.get_group(user)
        user_arms = records['item_id'].values
        
        for t in xrange(T):
            
            for algorithm,algorithm_obj in algorithms.items():
                if t == 0:
                    algorithm_obj.arm_list={i:linucb_arm(d,lambda_,alpha) for i in user_arms}
                #for each arm in the current user's arm_list
                for arm in algorithm_obj.arm_list:
                    algorithm_obj.inference_step(arm,item_features)
                max_arm = algorithm_obj.recommend()
                #remove the max arm after recommending it:
                
                response = records[records['item_id']==max_arm]['rating']
                response = int(response>4)
                algorithm_obj.updateParams(max_arm,response,item_features)
                precisions[algorithm][i] += (response/(1.0*len(users)))
                
                recalls[algorithm][i] += (response/(1.0*len(algorithm_obj.arm_list)*len(users)))
                del algorithm_obj.arm_list[max_arm]
    
    return precisions,recalls


def plot_function(x,y,title_,x_title,y_title):
    fig = plt.figure()
    plt.plot(x,y)
    plt.title(title_)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    fig.savefig(title_+'_'+y_title+'_.png')


def getResult(algorithm_name,T):
    x = [0.01,0.1,0.5,1,5,10,20]
    pr_list  = []
    re_list = []
    for alpha in x:
        pr,re = evaluateAlgorithms_MovieLens(T,test,item_features,1,alpha,1)
        pr_list.append(sum(pr[algorithm_name]))
        re_list.append(sum(re[algorithm_name]))
    print pr_list
    print re_list

    plot_function(x,pr_list,str(T),'alpha','cum_precision')
    plot_function(x,re_list,str(T),'alpha','cum_recall')


#getResult('linucb',120)
pr,re = evaluateAlgorithms_MovieLens(10,test,item_features,1,0.1,1)
print sum(pr['linucb'])
print sum(re['linucb'])


pr,re = evaluateAlgorithms_MovieLens(20,test,item_features,1,0.1,1)
print sum(pr['linucb'])
print sum(re['linucb'])

pr,re = evaluateAlgorithms_MovieLens(40,test,item_features,1,0.1,1)
print sum(pr['linucb'])
print sum(re['linucb'])

pr,re = evaluateAlgorithms_MovieLens(120,test,item_features,1,0.1,1)
print sum(pr['linucb'])
print sum(re['linucb'])


