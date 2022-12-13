# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:07:34 2022

@author: Yannick Wack
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class benchmarkNetworks:
    """
    A class that represents a scalable circular network of nodes and edges with producers and consumers. It can be used to generate the benchmark networks for the paper.
    """
    def __init__(self):
        """
        Initializes a new `benchmarkNetworks` object.
        """
        self.r0 = 100   # Initial radius of the network
        self.Nx = None  # Node x coordinates
        self.Ny = None  # Node y coordinates
        self.Es = None  # Source nodes of edges
        self.Et = None  # Target nodes of edges
        self.p = []     # List of producer nodes
        self.c = None   # List of consumer nodes
        self.j = None   # List of junction nodes
        self.NodeNumber = None # Number of nodes in the network
    # Build the two benchmark cases
    def buildOneProducerCase(self,segmentNumber):
        """
        Creates a circular network with one producer in the center and 'segmentNumber' segements around it.
        
        :param segmentNumber: The number of segments in the network.
        :type segmentNumber: int
        """
        self.addSegments(segmentNumber)
    def buildTwoProducerCase(self,circleNumber):
        """
        Creates a circular network with two producers on each side and 'circleNumber' rings around it.
        
        :param circleNumber: The number of rings in the network.
        :type circleNumber: int
        """
        self.addCircles(circleNumber)
    
    # Main methods building a network based on segeemnts and circles
    def addSegments(self, N):
        """
        Adds segments to the circular network.
        
        :param N: The number of segments to add.
        :type N: int
        """
        # Initial ring of nodes and edges 
        r = self.r0
        theta = [0, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5/4*np.pi, 6/4*np.pi, 7/4*np.pi]
        rho = [0, r, r, r, r, r, r, r, r]

        source = [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        target = [2, 4, 6, 8, 3, 4, 5, 6, 7, 8, 9, 2]

        r = r + self.r0
        l = 2  # ring number
        s = 1  # segment number
        id0 = [2, 3, 4, 5, 6, 7, 8, 9] # node ids of the previous ring

        # Add the additional segments to the network
        for k in range(N):
            n = 2**(l+2)
            n0 = 2**(l+1)  # number of nodes in previous ring
            sN = 4*2**(l-1)  # number of segments in ring

            # Make initial segment of ring
            if s == 1:
                theta.append(0)
                rho.append(r)
                source.append(id0[0])
                target.append(len(theta))

            # Final segment of ring
            if s == sN:
                theta.append((s/sN - 1/n)*2*np.pi)
                rho.append(r)
                source.extend([len(theta) - n+1, len(theta)])
                target.extend([len(theta), len(theta)-1])

                # Add cross edges between the 4 outer corners of the segment
                source.extend([len(theta) - n+1])
                target.extend([id0[s-1]])
                source.extend([id0[s-n]])
                target.extend([len(theta)-1])                

                # Jump to next ring
                s = 1
                l = l+1
                r = r + self.r0
                id0 = list(range(id0[-1]+1, id0[-1]+n+1))  # update node ids of the previous ring
                continue
                
            # Add nodes
            theta.extend([(s / sN - 1 / n) * 2 * np.pi, s / sN * 2 * np.pi])
            rho.extend([r, r])
            # Add Edges
            idL = len(theta) # Left node
            idR = idL - 1   # Right node
            # Angular Edges
            source.append(idL)
            target.append(idR)
            source.append(idR)
            target.append(idL - 2)
            # Radial edges
            source.append(idL)
            target.append(id0[s])
            # Add cross edges between the 4 outer corners of the segment
            source.append(idL - 2)
            target.append(id0[s])
            source.append(idL)
            target.append(id0[s-1])

            s = s + 1  # Iterate segment
        # Convert polar coordinates to Cartesian coordinates
        x, y = cartesian(theta, rho)

        # Check that source and target dont contain duplicate edges. If they do, remove them. Raise warning if duplicate edges are found.
        df = pd.DataFrame({'source': source, 'target': target})
        df = df.drop_duplicates()
        if len(df) != len(source):
            print('Duplicate edges found in network. Please check the source and target lists.')
        source = df['source'].tolist()
        target = df['target'].tolist()

        # Save coordinates and edge information to object properties
        self.NodeNumber = len(x)
        self.Nx = x
        self.Ny = y
        self.Es = source
        self.Et = target

        # Define connections and junctions in network
        self.defineConJun(centralProducer=True)

        # Append producer nodes to the network
        self.p = [0]
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

        # Append producer nodes to the list of nodes
        self.p = [self.NodeNumber-1, self.NodeNumber-2]
    
    # Submethods of the addCircles method
    def makeNodes(self, N):
        """
        Makes the nodes for the circular network.

        Arguments:
        N: The number of circles in the network.

        Returns:
        theta: The angular coordinates of the nodes.
        rho: The radial coordinates of the nodes.
        """
        theta = [0]
        rho = [0]
        r = 0
        for l in range(1, N + 1):   # l is the ring number
            r = r + self.r0
            n = 2**(l+2)
            for k in range(1, n + 1):   # k is the node number
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
    def defineConJun(self,centralProducer = False):
        """
        Define consumer and junction nodes in the current graph.

        Arguments:
        centralProducer: Boolean indicating whether the central producer node should be included in the list of consumers.



        """
        
        # Initialize empty lists for consumers and junctions
        self.c = []
        self.j = []
        a = 3
        
        # Iterate over all nodes in the graph
        for k in range(self.NodeNumber-2):
            
            # Skip the central producer node
            if centralProducer and k == 0:
                continue

            # Check if the current node is a consumer or a junction
            if k % a == 0:
                self.j.append(k)
            else:
                self.c.append(k)
    
    # Plot and save the network
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
        for i in range(self.NodeNumber):
            
            # Check if the current node is a consumer, junction, or producer
            if i in self.c:
                node_colors.append(consumer_color)
            elif i in self.j:
                node_colors.append(junction_color)
            elif i in self.p:
                node_colors.append(producer_color)
            else:
                node_colors.append("black")
            
        # Visualize the network
        plt.figure()
        plt.scatter(self.Nx, self.Ny, c=node_colors)
                
        for i in range(len(self.Es)):
            plt.plot([self.Nx[self.Es[i]-1], self.Nx[self.Et[i]-1]], [self.Ny[self.Es[i]-1], self.Ny[self.Et[i]-1]], 'k-')
        
        # Add an independent legend. Place the legend in the top left corner of the plot.
        plt.scatter([], [], c=consumer_color, label="Consumer")
        plt.scatter([], [], c=junction_color, label="Junction")
        plt.scatter([], [], c=producer_color, label="Producer")
        plt.legend(loc="upper left")

        
        
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
        
        # Create an output path one level above the current file
        outputPath = os.path.join(os.path.dirname(__file__), "..", "output")
        
        # Create the output folder if it does not exist
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        
        # Define the path to the output file
        outputFilePath = os.path.join(outputPath, name + ".xlsx")
        
        # Export the DataFrame to the output file
        df.to_excel(outputFilePath, index=False, sheet_name="Nodes")
        
        # Define the data for the "Edges" sheet
        ID = []
        Source = []
        Target = []
        
        # Iterate over all connections in the network
        for i in range(len(self.Es)):
            ID.append(i)
            Source.append(self.Es[i])
            Target.append(self.Et[i])
        
        # Create a dictionary from the data
        data = {"ID": ID, "Source": Source, "Target": Target}
        
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)
        
        # Add the DataFrame to the output file without losing the data in the "Nodes" sheet by using the ExcelWriter class
        with pd.ExcelWriter(outputFilePath, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, index=False, sheet_name="Edges")

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