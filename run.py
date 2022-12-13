# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:34:37 2022

@author: Yannick Wack (yannick.wack@kuleuven.be)

This file is used to run the network generator. It will generate the network
for the two benchmark cases and plot them and export them into the output folder.
"""
from src import networkGenerator as ng

# Create the circular network for the onoe producer benchmark case
oneProducerNetwork = ng.benchmarkNetworks()
oneProducerNetwork.buildOneProducerCase(12)     # Build the single producer network with 12 segments

oneProducerNetwork.plot()
oneProducerNetwork.export("oneProducerCase")

# Create the two producer benchmark case
twoProducerNetwork = ng.benchmarkNetworks()
twoProducerNetwork.buildTwoProducerCase(2)    # Build the two producer network with 2 circles

twoProducerNetwork.plot()
oneProducerNetwork.export("twoProducerCase")
