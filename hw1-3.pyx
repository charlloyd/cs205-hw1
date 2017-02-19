# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy

# test cases
sizes = [2**6, 2**10, 2**20, 2**32]
lists = [[1]*size for size in sizes] #each element is a list of specified size


def summmation(a):
    sums = a[0]
    i = 1
    N = len(a)
    while i < N:
        sums += a[i]
        i += 1
        
    return sums


