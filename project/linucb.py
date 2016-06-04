import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy import optimize

#Evaluation metrics: cumulative precision and cumulative recall
#computational complexity is linear in the number of arms and at most cubic in the number of features. 

class linucb_arm(object):
    def __init__(self,d,lambda_,alpha):
        self.d = d #dimension
        self.A = lambda_*np.identity(d)
        self.A_inv = inv(self.A)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.alpha = alpha
        self.p = 0
    
    def updateTheta(self):
        self.theta = np.dot(self.A_inv,self.b)
    
    def calcVariance(self,featureVec):
        return np.dot(np.dot(np.transpose(featureVec),self.A_inv),featureVec)
    
    def updateP(self,featureVec):
        self.p = np.dot(self.theta,featureVec)+self.alpha*np.sqrt(self.calcVariance(featureVec))

    def updateA(self,featureVec):
        self.A = self.A + np.outer(featureVec,featureVec)
        self.A_inv = inv(self.A)

    def updateb(self,featureVec,reward):
        
        self.b += reward*(featureVec)


class linucb(object):
    def __init__(self,arms,d,lambda_,alpha):
        self.d = d
        #each user has a linucb object
        self.arm_list = {i:linucb_arms(d,lambda_,alpha) for i in arms}
        self.lambda_ = lambda_
        self.alpha = alpha
        self.rewards =[]
        self.precision = 0 #w/o being normalized by num of users
        self.recall = 0
    
    
    def recommend(self):
        max_p = 0
        max_arm = self.arm_list.keys()[0]
        for arm,arm_obj in self.arm_list.items():
            if arm_obj.p>max_p:
                max_p = arm_obj.p
                max_arm = arm
    
        return max_arm

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
        






