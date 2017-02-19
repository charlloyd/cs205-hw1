# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def summmation(a):
    sums = a[0]
    i = 1
    N = len(a)
    while i < N:
        sums += a[i]
        i += 1
        
    return sums

def countpennies(n,ct):

    # first phase - everyone adds n times
    individual_count = 256 // n
    leftovers = 256 % n
    elapsed = individual_count - 1
    
    # middle phase - some people add leftovers and some add theirs together
    togethers = n - leftovers
    if leftovers!=0:
        togethers = (togethers // 2) + (togethers % 2) + leftovers
        elapsed += 1
        if ct==1:
            elapsed += 1

    # finish adding togethers
    while togethers > 1:
        togethers = (togethers // 2) + (togethers % 2)
        elapsed += 1
        if ct==1:
            elapsed += 1

    return elapsed

print('done')

n = list(range(1,150))
t = []

for i in n: 
    t.append(countpennies(i,0))
