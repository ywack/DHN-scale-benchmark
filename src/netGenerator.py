# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:00:08 2022

@author: Yannick Wack
"""
import numpy as np

class circNet:
    r0 = 100
    
    def __init__(self):
        self.p = 1 # One central producer
        
    def addSegments(self,N):
            ## addSegments Add N segments to the circle
            # Initial ring
            r = self.r0;
            theta = [0,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5/4*np.pi,6/4*np.pi,7/4*np.pi]
            rho = [0,r,r,r,r,r,r,r,r]
            
            source = [1,1,1,1,2,3,4,5,6,7,8,9]
            target = [2,4,6,8,3,4,5,6,7,8,9,2]
            
            r = r + self.r0
            l = 2  # Ring
            s = 1  # Segment
            id0 = [2,3,4,5,6,7,8,9]
            for k in range(N):
                n = 2**(l+2)
                n0 = 2**(l+1); # number of nodes in previous ring
                sN = 4*2**(l-1); # number of segments in ring
                
                # Make initial segment of ring
                if s == 1:
                    theta.append(0)
                    rho.append(r)
                    source.append(id0[1])
                    target.append(len(theta))
                    
                # Final segment of ring
                if s == sN:
                    theta.append((s/sN - 1/n)*2*np.pi)
                    rho.append(r)
                    source.extend([len(theta) - n+1,len(theta)])
                    target.extend([len(theta),len(theta)-1])
                    # Aditional cross segments
                    source.extend([len(theta) - n+1,len(theta)-1])
                    target.extend([id0[s],id0[s]-n0+1])
                    # Jump to next ring
                    s = 1
                    l = l+1
                    r = r + self.r0
                    id0 = range(id0[-1]+1,id0[-1]+n)
                    continue
                
                # Add nodes
                theta.extend([(s/sN - 1/n)*2*np.pi,s/sN*2*np.pi])
                rho.extend([r,r])
                
                # Add Edges
                idL = len(theta)
                idR = idL-1;
                # Angular Edges
                source.extend([idL,idR])
                target.extend([idR,idL-2])
                # Radial edges
                source.append(idL)
                target.append(id0[s+1])
                # Aditional cross segments
                source.append(idL)
                target.append(id0[s])
                source.append(idL-2)
                target.append(id0[s+1])
                
                s= s +1 # Iterate segment
            
            
            [x,y] = polToCart(theta,rho)
            
            # save
            self.NodeNumber = len(x)
            self.Nx = x
            self.Ny = y
            
            self.Es = source
            self.Et = target
            
            #self.defineConJun()

def polToCart(th,r):
    x = r*np.cos(th)
    y = r*np.sin(th)
    return(x,y)