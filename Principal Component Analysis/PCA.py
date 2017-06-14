# -*- coding: utf-8 -*-
# function :  pca function for data with row * column 
#             col : the number of data 
#             row : the number of features for each sample

# realize step :
#             1) normalize the data with data - mean( data ) for each sample 
    
#             2) Abstract the mean value for each column
    
#             3) Calculate the covariance matrix
  
#             4) Calculate the feature value

#             5) Calculate the feature vector
    
import numpy as np

def pca( data ):
    # Step : 1  For each sample X - mean(X )
    data_mean = np.mean(data, axis = 0)
    
    # Step : 2 Abstract the mean value for each column
    new_data = data - data_mean 
    
    # Step : 3 Calculate the covariance 
    X_X_T = np.dot( new_data, new_data.transpose() )
    
    # Step : 4 Calculate the feature value
    eig_value , eig_vector = np.linalg.eig( X_X_T )
    
    # Step : 5 Sort the eig_value
    index = np.argsort( eig_value )
    
    #
    print index
    print eig_value 
    print eig_vector
    
if __name__== '__main__':
    ################
   train_data = np.array([[2,3,2,5],[0,1,2,1]])
   pca( train_data )