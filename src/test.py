# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:34:37 2022

@author: u0128847
"""

# Import the circNet2Prod class
from networkGenerator import circNet2Prod

# Create a network with 1 circular layer
net = circNet2Prod()
net.addCircles(3)
# Visualize the network
net.plot()