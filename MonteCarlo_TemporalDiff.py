import pandas as pd
import bisect
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

'''
Note: gamma = 1 for everything
Brief intro: compare MC-constant-alpha(), MC-mean(), TD(0), forwardTD(lambda) and backwardTD(lambda)
Theoretical expectation: MC-const-alpha()==forwardTD(1)==backwardTD(1), TD(0)==forwardTD(0)==backwardTD(0) 
Results: proved the theoretical expectations (however backwardTD() might be slightly different based on how you implement 
eligibility trace)
Also, if you change constant alpha into k/T (total number of iterations), the values will converge. If a (learning rate) is too large, such as 0.5, the values wont converge.
'''

gamma = 1

graph = {1:[2],2:[3,4],3:[1,2,3,4],4:[]}
arcs = {(1,2):-2,(2,3):-2,(2,4):0,(3,1):1,(3,2):1,(3,3):1,(3,4):10}
#transitive distribution
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
#E5 = [(1,-2),(2,0),(4,0)]
E5 = [(1,-2),(2,-2),(3,1),(1,-2),(2,-2),(3,10),(4,0)]


#online update

def MC(index,epi,values,counts):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    for state,r in epi[:-1]:
        state -=1
        counts[state]+=1
        #values[index][state]+=(G-values[index][state])*1.0/counts[state]
        prev =values[state]
        values[state]=prev+(G-prev)*1.0/counts[state]
        G-=r
    return [v for k,v in values.items()]

#online update
def MC_nonstationary(index,epi,values,a):
    #calculate total sum from the begining
    G = sum(map(lambda x:x[1],epi))
    for state,r in epi[:-1]:
        state-=1
        values[state]+=(G-values[state])*a
        #returns[state]+=G
        G-=r
    return [v for k,v in values.items()]

def TD_0(index,epi,values,a):
    for i,e in enumerate(epi[:-1]):
        state = e[0]-1
        r = e[1]
        values[state] += a*(r+values[epi[i+1][0]-1]-values[state])
    return [v for k,v in values.items()]

#forward_TD
def TD_lambda(index,epi,values,lambda_,a):
    rewards = map(lambda x:x[1],epi)
    G_total= sum(rewards)
    T = len(rewards)
    for i,e in enumerate(epi[:-1]):
        #sum of returns
        state = e[0]-1
        r = e[1]
        G= 0
        for d in xrange(1,T-i):
            G+=(lambda_**(d-1))*(sum(rewards[i:i+d])+values[epi[i+d][0]-1])

        G =G*(1-lambda_)+G_total*lambda_**(T-i-1)
        values[state]+=a*(G-values[state])
        G_total-=r

    return [v for k,v in values.items()]


def TD_back_lambda(index,epi,values,lambda_,a):
    #memorize
    #values[index] = [i for i in values[index-1]]
    etrace = np.zeros(4)
    trace = []
    lambda_val = 1
    #for each state in one episode
    for i,e in enumerate(epi[:-1]):
        state = e[0]-1
        r = e[1]
        next_state = epi[i+1][0]-1
        error = r+values[next_state]-values[state]
        etrace[state]=1

        #for each state
        for s in xrange(4):
            values[s]+= a*error*etrace[s]
            etrace[s]*=lambda_
    
        #
    return [v for k,v in values.items()]



episodes = [E1,E2,E3,E4,E5]
episodes+=generateEpisodes(95,graph,arcs)
numepisodes=len(episodes)
#learning rate
a = 1.0/numepisodes

def plot_function(df,numepisodes,title_):
    
    title_ +='_episodes '+str(numepisodes)
    plot = df.plot(title=title_)
    plot.set_xlabel('episodes')
    plot.set_ylabel('values')
    plot.set_ylim([df.values.min()*1.1,df.values.max()*1.1])
    fig = plot.get_figure()
    fig.savefig(title_+'.png')



def run_MC_incremental():
    #values = [[0 for i in xrange(4)] for i in xrange(len(episodes))]
    #counts = [0 for i in xrange(4)]
    
    values = {i:0 for i in xrange(4)}
    value_matrix = []
    counts = np.zeros(4)
    returns= np.zeros(4)
    for index,epi in enumerate(episodes):
        value_matrix.append(MC(index,epi,values,counts))
    df = pd.DataFrame(value_matrix, columns=list('1234'))
    print 'MC_incremental'
    print df
    plot_function(df,numepisodes,'MC_incremental')

run_MC_incremental()


def run_MC_nonstationary(a):
    #values = [[0 for i in xrange(4)] for i in xrange(len(episodes))]
    #counts = [0 for i in xrange(4)]
    #values = np.zeros((len(episodes),4))
    values = {i:0 for i in xrange(4)}
    value_matrix = []
    for index,epi in enumerate(episodes):
        value_matrix.append(MC_nonstationary(index,epi,values,a))
    df = pd.DataFrame(value_matrix, columns=list('1234'))
    print df
    print 'MC_nonstationary'
    plot_function(df,numepisodes,'MC_nonstationary')

run_MC_nonstationary(a)




def run_TD(a):
    values = {i:0 for i in xrange(4)}
    value_matrix = []
    
    for index,epi in enumerate(episodes):
        value_matrix.append(TD_0(index,epi,values,a))

    df = pd.DataFrame(value_matrix, columns=list('1234'))
    print "TD 0"
    print df
    #plot
    plot_function(df,numepisodes,'TD(0)')

run_TD(a)



def forward_run_TD_lambda(lambda_,a):
    
    #values = np.zeros((len(episodes),4))
    values = {i:0 for i in xrange(4)}
    value_matrix = []
    for index,epi in enumerate(episodes):
        value_matrix.append(TD_lambda(index,epi,values,lambda_,a))
    
    df = pd.DataFrame(value_matrix, columns=list('1234'))
    print "forward TD",lambda_
    print lambda_,df
    #plot
    plot_function(df,numepisodes,'forward_TD'+'('+str(lambda_)+')')

forward_run_TD_lambda(0,a)
forward_run_TD_lambda(0.5,a)
forward_run_TD_lambda(1,a)

def backward_run_TD_lambda(lambda_,a):
    
    #values = np.zeros((len(episodes),4))
    values = {i:0 for i in xrange(4)}
    value_matrix = []
    for index,epi in enumerate(episodes):
        
        value_matrix.append(TD_back_lambda(index,epi,values,lambda_,a))
        #TD_lambda(index,epi,values,lambda_,a)
    df = pd.DataFrame(value_matrix, columns=list('1234'))
    print "backward TD",lambda_
    print df
    #plot
    plot_function(df,numepisodes,'backward_TD'+'('+str(lambda_)+')')

backward_run_TD_lambda(0,a)
backward_run_TD_lambda(0.5,a)
backward_run_TD_lambda(1,a)
