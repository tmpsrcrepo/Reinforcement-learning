import pandas as pd
import bisect
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

#wind strength for each column (# cells shifted upwords)
wind_strengths =[0,0,0,1,1,1,2,2,1,0]
ncol = len(wind_strengths)
nrow = 7
start = [3,0]
end = [3,7]
#initialize parameter
#episilon = 0.1


def createEmptyCanvas():
    #used for repsenting final policy
    empty_gridword = [[0 for j in xrange(ncol)] for i in xrange(nrow)]
    return empty_gridword

def initializeQ(nActions):
    Qvals = {} #q values, key: (s,a), val: value (s = i,j) (a=0,1,2,3 = left,up,right,down
    #4 possible ac
    for a in xrange(1,nActions+1):
        for i in xrange(nrow):
            for j in xrange(ncol):
                Qvals[(i,j,a)] = 0
    return Qvals



def findBestAction(i,j,actionSpace,Qvals):
    candidates = [0]*len(actionSpace)
    
    for k, (ai,aj) in actionSpace.items():
        out = (i,j,k)
        candidates[k-1] = Qvals[out]
    
    index_opt = np.argmax(candidates)
    value = candidates[index_opt]
    index_opt+=1
    a_i,a_j = actionSpace[index_opt]
    i_next = max(min(i+a_i,nrow-1),0)
    j_next = max(min(j+a_j,ncol-1),0)
    return value,index_opt,(i_next,j_next)

def epsilon_greedy(i,j,Qvals,actionSpace,episilon):
    nActions = len(actionSpace)
    cdf = [0]*(nActions+1)
    
    _,index_opt,candidates_pair = findBestAction(i,j,actionSpace,Qvals)
    
    for index in xrange(1,nActions+1):
        if index == index_opt:
            cdf[index] = cdf[index-1]+episilon*1.0/nActions+1-episilon
        else:
            cdf[index] = cdf[index-1]+episilon*1.0/nActions
    #make selection
    random_ = random.uniform(0,1)
    
    nex = bisect.bisect_left(cdf,random_)
    return nex,candidates_pair

def uniform_policy(i,j,Qvals,actionSpace):
    #make selection
    nex = random.sample(actionSpace.keys(),1)[0]
    a_i,a_j = actionSpace[nex]
    i_next = max(min(i+a_i,nrow-1),0)
    j_next = max(min(j+a_j,ncol-1),0)
    return nex,(i_next,j_next)


def plot_function(res,n_episodes,title_):
    title_ +='_episodes '+str(n_episodes)
    fig = plt.figure()
    plt.plot(res,np.arange(0,n_episodes+1))
    plt.title(title_)
    plt.ylabel('episodes')
    plt.xlabel('steps')
    fig.savefig(title_+'.png')




#SARSA
def Sarsa(n_episodes,alpha,nActions,actionSpace,episilon):
    sVals = createEmptyCanvas() #state values
    a_optimals = createEmptyCanvas()
    Qvals = initializeQ(nActions)
    
    res = [0] #episode index, #total number of steps to reach from
    i,j = start
    g_i,g_j = end
    print 'SARSA'
    #choose A
    for epi in xrange(n_episodes):
        a_optimals = createEmptyCanvas()
        nsteps = 0
        i,j = start
        a,candidate = epsilon_greedy(i,j,Qvals,actionSpace,episilon)
        while i!=g_i or j!= g_j:
            #take the action
            a_optimals[i][j] = a
            a_i,a_j = candidate
            a_i = max(0,a_i-wind_strengths[a_j])
            
            #find the next move
            a_next,candidate = epsilon_greedy(a_i,a_j,Qvals,actionSpace,episilon)
            #next_i,next_j =candidate
            #update Qvals
            val = Qvals[(i,j,a)]
            
            Qvals[(i,j,a)] =val + alpha*(-1+Qvals[(a_i,a_j,a_next)]-val)
            #update moves
            i,j =a_i,a_j
            a = a_next
            nsteps+=1
        
        res.append(nsteps+res[-1])
    
    plot_function(res,n_episodes,'SARSA_'+str(nActions))

    for i, row in enumerate(a_optimals):
        print (row)

#print a_optimals


#Q-learning
def q_learning(n_episodes,alpha,nActions,actionSpace,episilon):
    sVals = createEmptyCanvas() #state values
    #final_a_optimals = None
    #min_steps = None
    Qvals = initializeQ(nActions)
    print 'Q-learning'
    res = [0] #episode index, #total number of steps to reach from
    i,j = start
    g_i,g_j = end
    last = 0
    #choose A
    for epi in xrange(n_episodes):
        a_optimals = createEmptyCanvas()
        #nsteps = 0
        i,j = start
        #for each step of episode
        while i!=g_i or j!= g_j:
            #find the next move a from current position (episilon-greedy)
            A,candidate = epsilon_greedy(i,j,Qvals,actionSpace,episilon)
            
            #record the current action
            a_optimals[i][j] = A
            
            #take action
            next_i,next_j = candidate
            next_i = max(0,next_i-wind_strengths[next_j])
            #print next_i
            
            #get Q(s,a)
            val = Qvals[(i,j,A)]
            #find the best action from the next position
            Q_next,index_,candidate = findBestAction(next_i,next_j,actionSpace,Qvals)
            
            Qvals[(i,j,A)] = val + alpha*(-1+Q_next-val)
            #update moves
            
            i,j =next_i,next_j
            #nsteps+=1
            last+=1
        #last+=nsteps
        res.append(last)
    
    plot_function(res,n_episodes,'Q_Learning_'+str(nActions)+'_episilon_'+str(episilon))
    
    for i, row in enumerate(a_optimals):
        print (row)



def q_learning_uniform(n_episodes,alpha,nActions,actionSpace,episilon):
    sVals = createEmptyCanvas() #state values
    #final_a_optimals = None
    #min_steps = None
    Qvals = initializeQ(nActions)
    print 'Q-learning_unifrom'
    res = [0] #episode index, #total number of steps to reach from
    i,j = start
    g_i,g_j = end
    last = 0
    #choose A
    for epi in xrange(n_episodes):
        a_optimals = createEmptyCanvas()
        #nsteps = 0
        i,j = start
        print epi
        #for each step of episode
        while i!=g_i or j!= g_j:
            #find the next move a from current position (episilon-greedy)
            A,candidate = uniform_policy(i,j,Qvals,actionSpace)
            
            #record the current action
            a_optimals[i][j] = A
            
            #take action
            next_i,next_j = candidate
            next_i = max(0,next_i-wind_strengths[next_j])
            #print next_i
            
            #get Q(s,a)
            val = Qvals[(i,j,A)]
            #find the best action from the next position
            Q_next,index_,candidate = findBestAction(next_i,next_j,actionSpace,Qvals)
            
            Qvals[(i,j,A)] = val + alpha*(-1+Q_next-val)
            #update moves
            
            i,j =next_i,next_j
            #nsteps+=1
            last+=1
        #last+=nsteps
        res.append(last)
    
    plot_function(res,n_episodes,'Q_Learning_uniform'+str(nActions)+'_episilon_'+str(episilon))
    
    for i, row in enumerate(a_optimals):
        print (row)

def main():
    '''initialization of wind gridworld
        #Reinforcement Learning, p136, Example 6.5
        #gridword: 10 columns, 7 rows
        '''
    
    gridworld = [wind_strengths for i in xrange(nrow)]
    actionSpace = {1:(0,-1),2:(-1,0),3:(0,1),4:(1,0)}
    nActions = 4
    
    n_episodes = 170
    alpha = 0.5
    
    #undiscounted, gamma = 1
    #reward = -1 unless it reaches the goal
    
    #results
    # graph: episodes vs steps
    # final policy representation
    '''task a: results of sarsa & q-learning'''
    episilon = 0.1
    Sarsa(n_episodes,alpha,nActions,actionSpace,episilon)
    print
    q_learning(n_episodes,alpha,nActions,actionSpace,episilon)
    print
    '''Part 2''
        '' do off-policy TD learning using uniform policy on non-king's move'''
    print 'uniform policy' #this algorithm is extremely slow cuz it assigns the same prob to each action
    q_learning_uniform(n_episodes,alpha,nActions,actionSpace,episilon)
    print
    
    '''task b: king's moves are available'''
    actionSpace = {1:(0,-1),2:(-1,-1),3:(-1,0),4:(-1,1),5:(0,1),6:(1,1),7:(1,0),8:(1,-1)}
    nActions = 8
    Sarsa(n_episodes,alpha,nActions,actionSpace,episilon)
    print
    q_learning(n_episodes,alpha,nActions,actionSpace,episilon)







if __name__ == '__main__':
    main()
