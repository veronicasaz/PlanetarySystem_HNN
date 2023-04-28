"""
Created: September 2022
Last modified: October 2022 
Author: Veronica Saz Ulibarrena 
Description: additional plotting tools
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

# COLORS
color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered','saddlebrown', 'olivedrab', 'darkorange',  'saddlebrown', 'darkorange', 'red' ]
color3 = ['navy', 'lightgreen', 'darkorange']

# TRUNCATE
def trunc(values, decs=0):
    """
    trunc: truncate values
    INPUTS:
        values: value to be truncated
        decs: (int) number of decimals for output
    OUTPUTS:
        truncated value
    """
    return np.trunc(values*10**decs)/(10**decs)

class CustomTicker(LogFormatterSciNotation):
    """
    CustomTicker: create logarithmic tick location
    """
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)

