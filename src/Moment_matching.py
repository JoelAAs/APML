# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:55:55 2023

@author: mohko200
"""

import numpy as np
from scipy.stats import truncnorm

def mutiplyGauss(m1, s1, m2, s2):
    s = 1/(1/s1+1/s2)
    m = (m1/s1+m2/s2)*s
    return m, s


def divideGauss(m1, s1, m2, s2):
    m, s = mutiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):
    a_scaled , b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled , b_scaled , loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled , b_scaled , loc=m0, scale=np.sqrt(s0))
    return m, s



m1 = 25 # The mean of the prior s1
s1 = 3 # The variance of the prior s1
m2 = 25 # The mean of the prior s2
s2 = 3 # The variance of the prior s1
sv = 1.5 # The variance of p(t|x)
y0 = 1 # The measurement



mu2_m = m1 # mean of message
mu2_s = s1 # variance of message

mu4_m = m2 # mean of message
mu4_s = s2 # variance of message


for j in range(0, 10):
    mu5_m = (mu2_m * mu4_s**2 + mu4_m * mu2_s**2) / (mu4_s**2 + mu2_s**2)
    mu5_s = sv**2 + (mu2_s**2 * mu4_s**2) / (mu2_s**2 + mu4_s**2)
    
    
    
    if y0 == 1:
        a, b = 0, np.Inf
    else:
        a, b = np.NINF , 0
    
    pt_m , pt_s = truncGaussMM(a, b, mu5_m , mu5_s)
    
    
    
    mu8_m , mu8_s = divideGauss(pt_m , pt_s , mu5_m , mu5_s)
    
    
    
    mu9_m = mu8_m
    mu9_s = mu8_s + sv
    
    mu10_m = mu8_m
    mu10_s = mu8_s + sv
    
    
    px_m , px_s = mutiplyGauss(mu2_m , mu2_s , mu9_m , mu9_s)
    px_m , px_s = mutiplyGauss(mu4_m , mu4_s , mu10_m , mu10_s)

print(px_m)
print(px_s)


