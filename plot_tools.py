import numpy as np

# COLORS
color1 = ['navy', 'dodgerblue','darkorange']
color2 = ['dodgerblue', 'navy', 'orangered', 'green', 'olivedrab',  'saddlebrown', 'darkorange', 'red' ]

# TRUNCATE
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


