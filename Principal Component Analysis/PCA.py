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
    
#             6) Extract the max K eig_value
    
#             7) Extract the max K eig_vector corresponding to max K eig_value 
    
#             8) Project the samples to low-dimensional with w = eig_vector_K
    
#             9) Reduct the data   
    
import numpy as np

def pca( data , K ):
    
    if K > data.shape[1]:
         print 'K should be smaller than the dimensional of data'
         return
    # Step : 1  For each sample X - mean(X )
    data_mean = np.mean(data, axis = 0)
    
    # Step : 2 Abstract the mean value for each column
    new_data = data - data_mean 
    
    # Step : 3 Calculate the covariance 
    X_T_X = np.dot( new_data.transpose(), new_data )
    
    # Step : 4 Calculate the feature value
    eig_value , eig_vector = np.linalg.eig( X_T_X )
    
    # Step : 5 Sort the eig_value
    index = np.argsort( eig_value )
    
    # Step : 6 Extract the max K eig_value
    index_K = index[: -( K + 1 ): -1]
    
    # Step : 7 Extract the max K eig_vector corresponding to max K eig_value
    eig_vector_K = eig_vector[:, index_K]
    
    # Step : 8 Project the samples to low-dimensional with w = eig_vector_K
    low_dim_data = np.dot( data , eig_vector_K )
    
    # Step : 9 Reduct the data
    recon_Data = np.dot( low_dim_data , eig_vector_K.transpose() ) + data_mean
    
    return low_dim_data, recon_Data

    
if __name__== '__main__':
    ################
   train_data = np.array([[2,3,2,5],[0,1,2,1]])
   a , b = pca( train_data , 4)
   print a , b