import pandas as pd

import numpy as np
import random
import matplotlib.pyplot as plt


#Problem:  (intial 5 episodes are given, and then randomly generate future episodes)
graph = {1:[2],2:[3,4],3:[1,2,3,4],4:[]}
arcs = {(1,2):-2,(2,3):-2,(2,4):0,(3,1):1,(3,2):1,(3,3):1,(3,4):10}

#Find Monte-carlo (incremental,non-stationary), Temporal-Difference (TD(0), TD(1), forward 0.5,backward 0.5)
#a = 0.5, gamma = 1 (no decay)

def generateEpisodes(n,graph,arcs):
    out = []
    #start from 1 and end at 4
    for i in xrange(n):
        epi = []
        cur = 1
        nex = 0
        while cur!=4:
            nex = random.choice(graph[cur])
            epi.append((cur,arcs[(cur,nex)]))
            cur = nex
        out.append(epi+[(4,0)])
    return (out)


#episodes (initial 5)
E1 = [(1,-2),(2,0),(4,0)]
E2 = [(1,-2),(2,0),(4,0)]
E3 = [(1,-2),(2,-2),(3,1),(3,10),(4,0)]
E4 = [(1,-2),(2,-2),(3,10),(4,0)]
E5 = [(1,-2),(2,-2),(3,1),(1,-2),(2,-2),(3,10),(4,0)]


episodes = [E1,E2,E3,E4,E5]


def MC(index,epi,values,counts):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    returns= [0 for i in xrange(0,4)]
    for state,r in epi:
        counts[state-1]+=1
        returns[state-1]+=G
        prev =values[index-1][state-1]
        values[index][state-1]=prev+(returns[state-1]-prev)*1.0/counts[state-1]
        G-=r
    return values


def MC_nonstationary(index,epi,values,counts,a):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    returns= [0 for i in xrange(0,4)]
    for state,r in epi:
        counts[state-1]+=1
        returns[state-1]+=G
        prev =values[index-1][state-1]
        values[index][state-1]=prev+(returns[state-1]-prev)*a
        G-=r
    return values

def TD_0(index,epi,values,a):
    values[index] = [i for i in values[index-1]]
    for i,e in enumerate(epi[:-1]):
        state = e[0]-1
        r = e[1]
        values[index][state] += a*(r+values[index][epi[i+1][0]-1]-values[index][state])
    return values

#forward_TD
def TD_lambda(index,epi,values,lambda_,a):
    rewards = map(lambda x:x[1],epi)
    returns = np.zeros(4)
    values[index] = [i for i in values[index-1]]
    for i,e in enumerate(epi[:-1]):
        #sum of returns
        state = e[0]-1
        return_tmp  = 0
        for d in xrange(i+1,len(rewards)):
            return_tmp+=(lambda_**(d-i-1))*sum(rewards[i:d])
        values[index][state]+=a*((1-lambda_)*return_tmp-values[index][state])
    return values


def TD_back_lambda(index,epi,values,etrace,lambda_,a):
    #memorize
    values[index] = [i for i in values[index-1]]
    for i,e in enumerate(epi[:-1]):
        state = e[0]-1
        r = e[1]
        error = r+values[index][epi[i+1][0]-1]-values[index][state]
        etrace[state]+=1
        #update all the states
        for s in xrange(4):
            values[index][s] += a*error*etrace[s]
            etrace[s]*=0.5
    
    return values




def plot_function(df,numepisodes,title_):
    title_ +='_episodes '+str(5+numepisodes)
    plot = df.plot(title=title_)
    plot.set_xlabel('episodes')
    plot.set_ylabel('values')
    fig = plot.get_figure()
    fig.savefig(title_+'.png')



def run_MC_incremental():
    #values = [[0 for i in xrange(4)] for i in xrange(len(episodes))]
    #counts = [0 for i in xrange(4)]
    values = np.zeros((len(episodes),4))
    counts = np.zeros(4)
    for index,epi in enumerate(episodes):
        MC(index,epi,values,counts)
    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    plot_function(df,numepisodes,'MC_incremental')




def run_MC_nonstationary(a):
    #values = [[0 for i in xrange(4)] for i in xrange(len(episodes))]
    #counts = [0 for i in xrange(4)]
    values = np.zeros((len(episodes),4))
    counts = np.zeros(4)
    for index,epi in enumerate(episodes):
        MC_nonstationary(index,epi,values,counts,a)
    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    plot_function(df,numepisodes,'MC_nonstationary')






def run_TD(a):
    values = np.zeros((len(episodes),4))
    counts = np.zeros(4)
    
    for index,epi in enumerate(episodes):
        TD_0(index,epi,values,a)

    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    #plot
    plot_function(df,numepisodes,'TD(0)')





def forward_run_TD_lambda(lambda_,a):
    
    values = np.zeros((len(episodes),4))
    
    counts = np.zeros(4)
    for index,epi in enumerate(episodes):
        TD_lambda(index,epi,values,lambda_,a)
    
    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    #plot
    plot_function(df,numepisodes,'forward_TD'+'('+str(lambda_)+')')






def backward_run_TD_lambda(lambda_,a):
    
    values = np.zeros((len(episodes),4))
    
    counts = np.zeros(4)
    etrace = np.zeros(4)
    
    for index,epi in enumerate(episodes):
        TD_back_lambda(index,epi,values,etrace,lambda_,a)
        #TD_lambda(index,epi,values,lambda_,a)
    
    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    #plot
    plot_function(df,numepisodes,'backward_TD'+'('+str(lambda_)+')')






def main():
    numepisodes=open(sys.argv[1])
    episodes=[E1,E2,E3,E4,E5]+generateEpisodes(numepisodes,graph,arcs)
    run_MC_incremental()
    run_MC_nonstationary(0.5)
    run_TD(0.5)
    forward_run_TD_lambda(0.5,0.5)
    backward_run_TD_lambda(1,0.5)
    
    
if __name__ == '__main__':
    main()
