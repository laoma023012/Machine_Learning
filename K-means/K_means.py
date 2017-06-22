# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class K_means():
   def __init__( self , data , K ):
       # --------- data -----------
       self.data = data
       # -------- init K cluster -------
       self.K = K
   
   def euclDistance( self , vec_1 , vec_2 ):
       return np.sqrt( sum( np.power( vec_2 - vec_1 , 2 )) )
    
   def init_Centroids( self  ):
       # ---------- dimensional of data -----------------
       numSamples , dim = self.data.shape
       # --------- init the center ----------------
       centroids = np.zeros(( self.K , dim )) # row K --- col dim
       # -------- loop for init -------------------
       for index in range( self.K ):
           
           random_index = int( np.random.uniform( 0 , numSamples ) )
           centroids[index,:] = self.data[random_index,:]
       
       return centroids

   def kmeans( self ):
       
       numSamples = self.data.shape[0]
       clusterAssment = np.mat( np.zeros( (numSamples , 2 )) )
       clusterChanged = True
       
       # step - 1 init centroids
       centroids = self.init_Centroids( )
       
       while clusterChanged:
           
           clusterChanged = False
           # loop for every sample
           for index in xrange(numSamples):
               # --- minDist ----
               minDist = 100000
               minIndex = 0
               # step - 2 find the centroids who is closest
               for index_2 in range( self.K ):
                   dist = self.euclDistance( centroids[index_2,:] , self.data[index,:] )
                   if dist < minDist:
                       minDist = dist
                       minIndex = index_2
              
               # step - 3 update its cluster
               if clusterAssment[index,0] != minIndex:
                   clusterChanged = True
                   clusterAssment[index,:] = minIndex , minDist ** 2
                   
           # step - 4 update centroids 
           for index in range( self.K ):
               pointsInCluster = self.data[ np.nonzero(clusterAssment[:,0].A == index)[0] ]
               centroids[index,:] = np.mean( pointsInCluster , axis = 0 )
    
       print 'clustering process completed'
       return centroids , clusterAssment
       
   def show_cluster( self , centroids , clusterAssment ):
       numSamples , dim  = self.data.shape
       # ------- only plot for two -dimsional ------------
       if dim != 2:
           print ' Sorry , your k is too large '
           return 1
       
       mark_color = ['or' , 'ob' , 'og' , 'ok', '^r' , '+r' , 'sr' , 'dr' , '<r' , 'pr' ]
       
       if self.K > len(mark_color):
           print 'too large that I do not have enough color to draw it '
           return 1
       
       # draw all the samples
       for index in xrange(numSamples):
           markIndex = np.int(clusterAssment[index , 0])
           plt.plot(self.data[index , 0] , self.data[index , 1] , mark_color[markIndex])
           
       #------------------------------------------------------------------
       mark_color = ['Dr','Db','Dg','Dk','^b','+b','sb','db','<b','pb']
       #------- draw the centroids ---------------------------
       for index in range(self.K):
           plt.plot( centroids[index , 0] , centroids[index , 1] , mark_color[index] , markersize = 12 )
    
       plt.show()
              
if __name__=='__main__':
    
    data = np.array([[1,2],[2,3],[4,4],[6,6],[10,10],[16,3],[12,12],[22,15]])
    K = 3
    A = K_means(data,K)
    centroids , clusterAssment = A.kmeans()
    A.show_cluster(centroids , clusterAssment)