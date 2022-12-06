# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:07:34 2022

@author: u0128847
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class circNet2Prod:
    """
    A class that represents a circular network of nodes with producers and consumers.
    """
    def __init__(self):
        """
        Initializes a new `circNet2Prod` object.
        """
        self.r0 = 100
        self.Nx = None
        self.Ny = None
        self.Es = None
        self.Et = None
        self.p = []
        self.c = None
        self.j = None
        self.mapNode2jcp = None
        self.NodeNumber = None

    def addCircles(self, N):
        """
        Add circles of nodes and edges to the current graph.
        
        Args:
            N (int): The number of circles to add to the graph.
        """
        
        # Make nodes
        theta, rho = self.makeNodes(N)
    
        # Make edges
        s, t = self.makeEdges(N, theta, rho)
    
        # Add producer nodes
        theta, rho = self.addProducerNodes(N, theta, rho)
    
        # Transform to cartesian
        x, y = cartesian(theta, rho)
    
        # Connect to leftmost and rightmost consumers
        s_out, t_out = self.connectToConsumers(x)
    
        # Add producer connections to the list of edges
        s.extend(s_out)
        t.extend(t_out)
    
        # Save variables
        self.NodeNumber = len(theta)
        self.Nx, self.Ny = x, y
    
        self.Es, self.Et = s, t
        
        # Define consumer and junctions
        self.defineConJun()

    def makeNodes(self, N):
        theta = [0]
        rho = [0]
        r = 0
        for l in range(1, N + 1):
            r = r + self.r0
            n = 2**(l+2)
            for k in range(1, n + 1):
                theta.append(2*np.pi*k/n)
                rho.append(r)
        return theta, rho    
    def makeEdges(self, N, theta, rho):
        """
        Makes the edges for the circular network.
        
        Arguments:
        N: The number of circles in the network.
        theta: The angular coordinates of the nodes.
        rho: The radial coordinates of the nodes.
        
        Returns:
        s: The starting node for each edge.
        t: The ending node for each edge.
        """
        nodeCount = 1
        s = [1,1,1,1]  # List of starting nodes for the edges
        t = [3,5,7,9]  # List of ending nodes for the edges
        pN0 = []  # List of previous nodes for radial connections
        
        # Loop through each circle in the network
        for l in range(1, N+1):
            n = 2**(l+2)  # Number of nodes in the current circle
            pN = []  # List of current nodes for radial connections
            m = 1  # Index for pN0 list
            
            # Loop through each node in the circle
            for k in range(1, n+1):
                # Make angular connections
                if k == n:
                    s.append(nodeCount+k)
                    t.append(nodeCount+1)
                else:
                    s.append(nodeCount+k)
                    t.append(nodeCount+k+1)
                    
                # Make radial connections
                # Save current node for radial connections
                pN.append(nodeCount+k)
                if k % 2 == 0 and not l==1:  # Only make connections for even-numbered nodes, except for the first circle
                    s.append(nodeCount+k)
                    t.append(pN0[m-1])
                    
                    # Make additional cross connections
                    if k == n:
                        s.extend([nodeCount+k,nodeCount+2])  # nodeCount+k+2
                        t.extend([pN0[m],pN0[m-1]])  # pN0[m]
                    else:
                        s.extend([nodeCount+k,nodeCount+k+2])  # nodeCount+k+2
                        t.extend([pN0[m],pN0[m-1]])  # pN0[m]
                        
                    m += 1
        
            # Update pN0 list for the next circle
            pN0 = [pN[i] for i in range(len(pN))] + [pN[0]]
            nodeCount += n
        
        return [s, t]



    def addProducerNodes(self, N, theta, rho):
        """
        Adds two producer nodes to the network, at angles 0 and pi, and at a distance
        `r + r0/2` from the center of the network, where `r` is the radius of the
        outermost circle.
        """
        r = rho[-1]
        theta.extend([0, np.pi])
        rho.extend([r + self.r0/2, r + self.r0/2])
        return theta, rho
    
    def connectToConsumers(self, x):
        """
        Connect the leftmost and rightmost producers in the list x to the 3 smallest and 3 largest elements of x, respectively.
        
        Args:
            x (list): A list of integers.
        
        Returns:
            tuple: A tuple containing two lists of integers, representing the source and target indices of the connections.
        """
        
        # Find the leftmost and rightmost producers
        left_producer = min(x)
        right_producer = max(x)
    
        # Create a list of tuples containing the index and value of each element in x
        indexed_x = list(enumerate(x))
    
        # Sort the list of tuples by the value of each tuple
        sorted_x = sorted(indexed_x, key=lambda t: t[1])
    
        # Create a list of the indices of the 3 smallest elements of x, including duplicates
        smallest_indices = [i[0] for i in sorted_x[1:4]]
    
        # Create a list of the indices of the 3 biggest elements of x, including duplicates
        biggest_indices = [i[0] for i in sorted_x[-4:-1]]
    
        # Connect the leftmost producer to the 3 smallest elements
        s_out = [x.index(left_producer)] * 3
        t_out = smallest_indices
    
        # Connect the rightmost producer to the 3 largest elements
        s_out += [x.index(right_producer)] * 3
        t_out += biggest_indices
    
        # Add matlab offset
        t_out = [x + 1 for x in t_out]
        s_out = [x + 1 for x in s_out]
    
        return s_out, t_out


    def defineConJun(self):
        """
        Define consumer and junction nodes in the current graph.
        """
        
        # Initialize empty lists for consumers and junctions
        self.c = []
        self.j = []
        n2jcp = []
        a = 3
        
        # Iterate over all nodes in the graph
        for k in range(self.NodeNumber-2):
            
            # Check if the current node is a consumer or a junction
            if k % a == 0:
                self.j.append(k)
                n2jcp.append(f"Junction{len(self.j)}")
            else:
                self.c.append(k)
                n2jcp.append(f"Consumer{len(self.c)}")
        
        # Append producer nodes to the list of nodes
        self.p = [self.NodeNumber, self.NodeNumber-1]

    def plot(self):
        """
        Visualizes the network using Matplotlib.
        """
        
        # Set colors for consumers, junctions, and producers
        consumer_color = "blue"
        junction_color = "green"
        producer_color = "red"
        
        # Initialize list of colors for each node in the network
        node_colors = []        
        
        # Iterate over all nodes in the network
        for i in range(self.NodeNumber+1):
            
            # Check if the current node is a consumer, junction, or producer
            if i in self.c:
                node_colors.append(consumer_color)
            elif i in self.j:
                node_colors.append(junction_color)
            elif i in self.p:
                node_colors.append(producer_color)
            
        # Visualize the network
        plt.figure()
        plt.scatter(self.Nx, self.Ny, c=node_colors)
                
        for i in range(len(self.Es)):
            plt.plot([self.Nx[self.Es[i]-1], self.Nx[self.Et[i]-1]], [self.Ny[self.Es[i]-1], self.Ny[self.Et[i]-1]], 'k-')
        plt.show()
    def export(self, name):
        # Define the data for the "Nodes" sheet
        ID = []
        Type = []
        Xposition = []
        Yposition = []
        
        # Iterate over producers, consumers, and junctions
        for node_type, nodes in (("producer", self.p), ("consumer", self.c), ("junction", self.j)):
            for k in range(len(nodes)):
                ID.append(nodes[k])
                Type.append(node_type)
                Xposition.append(self.Nx[nodes[k]-1])
                Yposition.append(self.Ny[nodes[k]-1])
        
        # Create a dictionary from the data
        data = {"ID": ID, "Type": Type, "Xposition": Xposition, "Yposition": Yposition}
        
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)
        
        # Export the DataFrame to an Excel spreadsheet
        df.to_excel(f"topo_{name}_nodes_{len(self.Nx)}.xlsx", sheet_name="Nodes", index=False)

def cartesian(theta, rho):
    """
    Transforms polar coordinates to cartesian coordinates.
    
    Args:
        theta: A list of angles in radians.
        rho: A list of distances from the origin.
        
    Returns:
        x, y: The cartesian coordinates corresponding to the input polar coordinates.
    """
    x = [rho[i] * np.cos(theta[i]) for i in range(len(theta))]
    y = [rho[i] * np.sin(theta[i]) for i in range(len(theta))]
    return x, y