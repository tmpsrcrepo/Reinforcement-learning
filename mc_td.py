import pandas as pd
import bisect
import numpy as np
import random
import matplotlib.pyplot as plt

#gamma = 1 for everything


graph = {1:[2],2:[3,4],3:[1,2,3,4],4:[]}
arcs = {(1,2):-2,(2,3):-2,(2,4):0,(3,1):1,(3,2):1,(3,3):1,(3,4):10}
pdf = {1:[1],2:[0.8,1],3:[0.08,0.24,0.4,1],4:[]}
#P(3|3)=0.4, P(2|3)=0.4,P(1|3)=0.2


def generateEpisodes(n,graph,arcs):
    out = []
    #start from 1 and end at 4
    for i in xrange(n):
        epi = []
        cur = 1
        nex = 0
        while cur!=4:
            #nex = random.choice(graph[cur])
            random_ = random.uniform(0,1)
            cur_pdf = pdf[cur]
            nex = graph[cur][bisect.bisect_left(cur_pdf,random_)]
            epi.append((cur,arcs[(cur,nex)]))
            cur = nex
        out.append(epi+[(4,0)])
    return (out)


#episodes (initial 5)
E1 = [(1,-2),(2,0),(4,0)]
E2 = [(1,-2),(2,0),(4,0)]
E3 = [(1,-2),(2,-2),(3,1),(3,10),(4,0)]
E4 = [(1,-2),(2,-2),(3,10),(4,0)]
E5 = [(1,-2),(2,0),(4,0)]
E6 = [(1,-2),(2,-2),(3,1),(1,-2),(2,-2),(3,10),(4,0)]


#online update
#episodes = [E1,E2,E3,E4,E5,E6]
def MC(index,epi,values,counts):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    returns= [0 for i in xrange(0,4)]
    values[index] = [i for i in values[index-1]]
    for state,r in epi:
        state -=1
        counts[state]+=1
        #values[index][state]+=(G-values[index][state])*1.0/counts[state]
        returns[state]=G
        prev =values[index][state]
        values[index][state]=prev+(returns[state]-prev)*1.0/counts[state]
        G-=r
    return values

#online update
def MC_nonstationary(index,epi,values,counts,a):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    returns= [0 for i in xrange(0,4)]
    values[index] = [i for i in values[index-1]]
    for state,r in epi:
        state-=1
        #values[index][state]+=(G-values[index][state])*a
        returns[state]=G
        prev =values[index][state]
        values[index][state]+=(returns[state]-prev)*a
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
    #state_seq = map(lambda x:x[0]-1,epi)
    lambda_val = 1
    #for each state in one episode
    for i,e in enumerate(epi[:-1]):
        state = e[0]-1
        r = e[1]
        next_state = epi[i+1][0]-1
        prev = values[index][state]
        error = r+values[index][next_state]-prev
        etrace[state]+=1
        
        values[index][state] += a*error*etrace[state]*lambda_val
        lambda_val *=lambda_
        #etrace[state]*=lambda_
        #for s in xrange(4):
        #    values[index][s] += a*error*etrace[s]
        #    etrace[s]*=lambda_

    return values


numepisodes=94
episodes=[E1,E2,E3,E4,E5,E6]+generateEpisodes(numepisodes,graph,arcs)

def plot_function(df,numepisodes,title_):
    
    title_ +='_episodes '+str(6+numepisodes)
    plot = df.plot(title=title_)
    plot.set_xlabel('episodes')
    plot.set_ylabel('values')
    plot.set_ylim([df.values.min()*1.1,df.values.max()*1.1])
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
    print 'MC_incremental'
    print df
    plot_function(df,numepisodes,'MC_incremental')

run_MC_incremental()


def run_MC_nonstationary(a):
    #values = [[0 for i in xrange(4)] for i in xrange(len(episodes))]
    #counts = [0 for i in xrange(4)]
    values = np.zeros((len(episodes),4))
    counts = np.zeros(4)
    for index,epi in enumerate(episodes):
        MC_nonstationary(index,epi,values,counts,a)
    df = pd.DataFrame(values, columns=list('1234'))
    print df
    plot_function(df,numepisodes,'MC_nonstationary')

run_MC_nonstationary(0.5)




def run_TD(a):
    values = np.zeros((len(episodes),4))
    counts = np.zeros(4)
    
    for index,epi in enumerate(episodes):
        TD_0(index,epi,values,a)

    df = pd.DataFrame(values, columns=list('1234'))
    #print df
    #plot
    plot_function(df,numepisodes,'TD(0)')

run_TD(0.5)



def forward_run_TD_lambda(lambda_,a):
    
    values = np.zeros((len(episodes),4))
    
    counts = np.zeros(4)
    for index,epi in enumerate(episodes):
        TD_lambda(index,epi,values,lambda_,a)
    
    df = pd.DataFrame(values, columns=list('1234'))
    print lambda_,df
    #plot
    plot_function(df,numepisodes,'forward_TD'+'('+str(lambda_)+')')

forward_run_TD_lambda(0,0.5)



def backward_run_TD_lambda(lambda_,a):
    
    values = np.zeros((len(episodes),4))
    
    for index,epi in enumerate(episodes):
        etrace = np.zeros(4)
        TD_back_lambda(index,epi,values,etrace,lambda_,a)
        #TD_lambda(index,epi,values,lambda_,a)
    df = pd.DataFrame(values, columns=list('1234'))
    print df
    #plot
    plot_function(df,numepisodes,'backward_TD'+'('+str(lambda_)+')')


backward_run_TD_lambda(1,0.5)
#backward_run_TD_lambda(0.5,0.5)

