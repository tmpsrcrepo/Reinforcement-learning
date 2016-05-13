import numpy as np
import random
from numpy.linalg import inv


class linEps_arm(object):
    def __init__(self,d,lambda_):
        self.d = d #dimension
        self.A = lambda_*np.identity(d)
        self.A_inv = inv(self.A)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.p = 0
    
    def updateTheta(self):
        self.theta = np.dot(self.A_inv,self.b)
        print self.theta
    
    def updateP(self,featureVec):
        self.p = np.dot(self.theta,featureVec)
        print self.p
    
    def updateA(self,featureVec):
        self.A = self.A + np.outer(featureVec,featureVec)
        self.A_inv = inv(self.A)
    
    def updateb(self,featureVec,reward):
        self.b += reward*(featureVec)


class linEps(object):
    def __init__(self,arms,d,lambda_,epsilon):
        self.d = d
        #each user has a linucb object
        self.arm_list = {i:linEps_arm(d,lambda_) for i in arms}
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.rewards =[]
        self.precision = 0 #w/o being normalized by num of users
        self.recall = 0

    
    def recommend(self):
        
        random_ = random.uniform(0,1)
        if random_ <= 1-self.epsilon:
            max_p = 0
            max_arm = self.arm_list.keys()[0]
            for arm,arm_obj in self.arm_list.items():
                if arm_obj.p>max_p:
                    
                    max_p = arm_obj.p
                    max_arm = arm
            return max_arm
        else:
            #print np.random.choice(self.arm_list.keys(),1)
            return np.random.choice(self.arm_list.keys(),1)[0]

    def inference_step(self,arm,featureMatrix):
        featureVec = np.array(featureMatrix[:,arm-1].flatten())[0]
        #if arm not in self.arm_list:
        #    self.arm_list[arm] = linucb_arm(self.d,self.lambda_,self.alpha)
        self.arm_list[arm].updateTheta()
        self.arm_list[arm].updateP(featureVec)
    
    def updateParams(self,arm,response,featureMatrix):
        featureVec = np.array(featureMatrix[:,arm-1].flatten())[0]
        
        self.arm_list[arm].updateA(featureVec)
        
        self.rewards.append(response)
        self.arm_list[arm].updateb(featureVec,response)
