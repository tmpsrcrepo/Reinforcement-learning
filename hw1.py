#problem: find a fair oil leasing price over three years
# MDP, DP solution

import numpy as np
def solve(k): #k_states
    oil_price = 0.5*(20+30)
    cost = [0,130000,300000]
    pumped = [0,0.2,0.36]
    total = 100000
    
    dp_matrix = ([[(0,0) for j in xrange(k)] for i in xrange(3)])

    #i*j: i = 3 choices, j = k states
    
    #initialization: (amount of oil, total value)
    
    for i in xrange(3):
        #initial price = 20
        dp_matrix[i][0] = (total*(1-pumped[i]),20*total*pumped[i]-cost[i])
    print [row[0] for row in dp_matrix]
    
    last = max_val = 0
    for j in xrange(1,k):
        #print 'last',last
        for i in xrange(3):
            
            values = np.array([(dp_matrix[k][j-1])[1]+dp_matrix[k][j-1][0]*oil_price*pumped[i]-cost[i] for k in xrange(3)])
            
            state_index = values.argmax(axis = 0)
        
            dp_matrix[i][j] = (dp_matrix[state_index][j-1][0]*(1-pumped[i]),values[state_index])
            max_val = max(max_val,values[state_index])
        #print max_val
        #if the max value is no different than the max_value of the last iteration
        print [row[j] for row in dp_matrix]
        if last <= max_val:
            return max_val
        last = max_val





print solve(3)


