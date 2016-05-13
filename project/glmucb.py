import pandas as pd
import scipy.optimize
import numpy as np
from numpy.linalg import inv
import math

class glmucb_arm(object):
    def __init__(self,d,c,lambda_,link=lambda x: 1/(math.exp(-1*x)+1)):
        self.d = d #dimension
        self.c = c
        self.link = link
        self.A = lambda_*np.identity(d)
        self.A_inv = inv(self.A)
        self.gt = np.zeros(d)
        self.gt_ = np.zeros(d)
        self.theta = np.zeros(d)
        self.rewards = []
        self.features = []
        self.t = 0
        self.p = 0
    
    def updateGt(self,theta):
        featureVec = self.features[self.t]
        reward = self.rewards[self.t]
        self.gt_+= self.link(np.inner(featureVec,theta))*featureVec
        self.gt += (reward)
        return self.gt - self.gt_
    
    def updateTheta(self):
        theta = scipy.optimize.root(self.updateGt,self.theta).x
        self.theta = theta
        return theta
    
    def calcVariance(self,featureVec):
        return np.dot(np.dot(np.transpose(featureVec),self.A_inv),featureVec)
    
    def updateP(self):
        featureVec = self.features[self.t]
        t = self.t
        self.p = self.link(np.dot(self.theta,featureVec))+self.c*np.sqrt(math.log(t+1))*np.sqrt(self.calcVariance(featureVec))
    
    def updateA(self,featureVec):
        self.A += np.outer(featureVec,featureVec)
        self.A_inv = inv(self.A)

class glmucb(object):
    def __init__(self,arms,T,c,d,lambda_):
        self.arm_list = {i:glmucb_arm(d,c,lambda_) for i in arms}
        self.d = d
        self.c = c
        self.visited = False
        self.lambda_ = lambda_
    
    def recommend(self):
        max_p = 0
        max_arm = self.arm_list.keys()[0]
        for arm,arm_obj in self.arm_list.items():
            if not self.visited and arm_obj.p>max_p:
                max_p = arm_obj.p
                max_arm = arm

        return max_arm
    
    def inference_step(self,arm,featureMatrix,t,response):
        if t==0:
            return
        featureVec = np.array(featureMatrix[:,arm-1].flatten())[0]
        self.t = t
        self.arm_list[arm].features.append(featureVec)
        self.arm_list[arm].rewards.append(response)
        self.arm_list[arm].updateTheta()
        self.arm_list[arm].updateP()
    
    def updateParams(self,arm,response,featureMatrix):
        featureVec = np.array(featureMatrix[:,arm-1].flatten())[0]
        
        #self.arm_list[arm].rewards.append(response)
        
        self.arm_list[arm].updateA(featureVec)

