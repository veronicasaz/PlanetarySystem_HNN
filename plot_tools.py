import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

# COLORS
color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered', 'saddlebrown', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]

# TRUNCATE
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)

