# don't forget to "module load legacy/0.0.1-fasrc01; module load centos6/pandas-0.11.0_python-2.7.3" in shell

import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

sums = pd.read_csv('summation.csv')
matvec = pd.read_csv('matvec.csv')

y_sum = ['2^6','2^10','2^20']
y_matvec = ['2^6','2^10']

print(sums)
print(matvec)
