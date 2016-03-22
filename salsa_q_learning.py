import pandas as pd
import bisect
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

'''initialization of wind gridworld
#Reinforcement Learning, p136, Example 6.5
#gridword: 10 columns, 7 rows
'''
#wind strength for each column (# cells shifted upwords)
wind_strengths =[0,0,0,1,1,1,2,2,1,0]
ncol = len(wind_strengths)
nrow = 7
gridworld = [wind_strengths for i in xrange(nrow)]
actionSpace = {1:(0,-1),2:(-1,0),3:(0,1),4:(1,0)}
nActions = 4


#undiscounted, gamma = 1
#reward = -1 unless it reaches the goal
start = [3,0]
end = [3,7]

def createEmptyCanvas():
    #used for repsenting final policy
    empty_gridword = [[0 for j in xrange(ncol)] for i in xrange(nrow)]
    return empty_gridword

def initializeQ():
    Qvals = {} #q values, key: (s,a), val: value (s = i,j) (a=0,1,2,3 = left,up,right,down
        #4 possible ac
    for a in xrange(1,5):
        for i in xrange(nrow):
            for j in xrange(ncol):
                Qvals[(i,j,a)] = 0
    return Qvals


def epsilon_greedy(i,j,Qvals,wind_):
    cdf = [0]*(nActions+1)
    candidates = []
    candidates_pairs = []
    
    for k,(i_tmp,j_tmp) in actionSpace.items():
        i_next = i+i_tmp-wind_
        j_next = j+j_tmp
        if j_next == ncol:
            j_next-=1
        if i_next == nrow:
            i_next-=1
        out = (max(i_next,0),max(j_next,0),k)
        candidates+=[Qvals[out]]
        candidates_pairs+=[[out[0],out[1]]]
    #candidates = [ for k,(i_tmp,j_tmp) in (actionSpace.items())]
    #greedy move
    index_opt = np.argmax(candidates)+1
    
    for index in xrange(1,nActions+1):
        if index == index_opt:
            cdf[index] = cdf[index-1]+episilon*1.0/nActions+1-episilon
        else:
            cdf[index] = cdf[index-1]+episilon*1.0/nActions
    #make selection
    random_ = random.uniform(0,1)
    nex = bisect.bisect_left(cdf,random_)
    return nex,candidates_pairs

def plot_function(res,n_episodes,title_):
    title_ +='_episodes '+str(n_episodes)
    fig = plt.figure()
    plt.plot(res,np.arange(0,n_episodes+1))
    plt.title(title_)
    plt.ylabel('episodes')
    plt.xlabel('steps')
    fig.savefig(title_+'.png')


'''Part 1'''
#initialize parameter
episilon = 0.1

#SARSA
def Sarsa(n_episodes,alpha):
    sVals = createEmptyCanvas() #state values
    a_optimals = createEmptyCanvas()
    Qvals = initializeQ()
    
    res = [0] #episode index, #total number of steps to reach from
    i,j = start
    g_i,g_j = end
    #choose A
    for epi in xrange(n_episodes):
        nsteps = 0
        i,j = start
        a,candidates = epsilon_greedy(i,j,Qvals,wind_strengths[j])
        while i!=g_i or j!= g_j:
            #take the action
            a_optimals[i][j] = a
            a_i,a_j = candidates[a-1]
            
            #find the next move
            a_next,candidates = epsilon_greedy(i,j,Qvals,wind_strengths[j])
            next_i,next_j =candidates[a_next-1]
            #update Qvals
            val = Qvals[(i,j,a)]
            
            Qvals[(i,j,a)] =val + alpha*(-1+Qvals[(next_i,next_j,a_next)]-val)
            #update moves
            i,j =next_i,next_j
            a = a_next
            nsteps+=1
    
        res.append(nsteps+res[-1])
    
    plot_function(res,n_episodes,'SARSA')


    #print a_optimals

Sarsa(170,0.5)

#Q-learning



#results
# graph: episodes vs steps
# final policy representation

'''task a: results of sarsa & q-learning'''
'''task b: king's moves are available'''


'''Part 2''
'' do off-policy TD learning using uniform policy on non-king's move'''
