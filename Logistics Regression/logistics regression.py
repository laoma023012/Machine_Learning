# -*- coding: utf-8 -*-

# GradAscent with full data

#  alpha : learning rate
#  max_iteration : max itertaions with full loop

import numpy as np

class sigmoid():
    
  def __init__(self ,  data , label , max_iteration , alpha ): 
      self.data = data
      self.label = label 
      self.max_iteration = max_iteration
      self.alpha = alpha
      self.weight = None
  
  def sigmoid ( self , x ):
     
      num =  1 / (1 + np.exp( -1 * x ))    
     
      return num  
  #def sigmoid( self , inX ):
    
   # num =  0.5 * ( 1 + np.tanh( 0.5 * inX ) )
    
    #return num
    
  def GradAscent( self ):
    # convert data to matrix
    Data = np.mat( self.data )
    
    # convert label to matrix
    Label = np.mat( self.label ).transpose()
    #print 'Label',Label
    
    # calculate row, col for data
    row , col = data.shape
    
    # init - weights
    weights = np.mat( np.ones( col ) ).transpose()
    
    # loop for gradAscent
    for index in range( self.max_iteration ):
        
        h = self.sigmoid( np.dot( Data , weights ) )
        
        #print 'h = ',h
        
        Error = Label - h
        #print 'Error',Error
        
        weights = weights + self.alpha * Data.transpose() * Error 
    
    self.weight = weights
    
    #print 'weights',weights
    return self.weight

  def predict( self , data):
      
      label = self.sigmoid( np.dot( data , self.weight ) )
      #label = np.sign( label )
      return label
      
  def train( self ):

    self.GradAscent()
     
     
if __name__=='__main__':
    
    data = np.array([[1,2],[2,4],[3,3],[5,5],[5,6],[6,6],[8,5],[9,10],[10,14]])
    label = np.array([0, 0 , 0 , 0, 1 , 1 , 1 , 1, 1 ])
    
    max_iterations = 500
    alpha = 0.002
    
    A = sigmoid( data , label , max_iterations , alpha )
    A.train()
    
    label = A.predict( data )
    print label
