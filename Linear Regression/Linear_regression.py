# -*- coding: utf-8 -*-
## function : Linear regression 
## Description : This function describes the linear regression based on
##               Machine Learning Yujian

## Input : Data   The input data should be in row * column
##                column : represent the number of samples
##                row : represent the dimensional of sampkes

##         Label  The label should be in the form of row * column( usually = 1)
##                row : represent the number of samples

## Function : w = (X * X ^ T + lamada * I ) ^ -1 * X * Y

## Key word : You should add a row with vector ones on the first line

## Output : w = [b ,w1 , w2, ... , wn ]

import numpy as np
from sklearn import linear_model

def data_standard( data , label ):

    ## judge whether the label corresponding to data    
    if data.shape[1] != label.shape[0]:
       print "The column number of data should be equal to the row number of label" 
       exit()
       #return 
    return data , label

## input : data 
##              row - dimensional of feature
##              col - number of samples
## output : w
##              w = [ b , w1 , w2 , ... , w_n ] 

def train( data , label , lamada = 1e-8 ):
    
    ## Judge whether the input data is illegal or not
    Data = np.array( data )
    label = np.array( label ) 
    Data , label = data_standard( data , label )
    
    ## Key Point : data must in the form of row * col  where col stand for
    ## the number of samples
    
    
##  Step:1 add a line with ones    
    Data = np.array(data)
    
## 构造一个全 1的 行向量 其中 列 等于  矩阵的列的维度     
    First_line = np.array(np.ones(data.shape[1]))
    # 加到 X 矩阵的第一行
    X = np.vstack(( First_line , Data ))
    
##  Step:2   计算 X 的转置
    X_T = X.transpose()
    
##  Step:3   X * X ^ T + lamada * I
    Func_1 = np.dot( X , X_T ) + lamada * np.eye( X.shape[0] )

##  Step:4  judge X * X ^ T 是否 可逆
    if np.linalg.det( Func_1 ) == 0 :
        print 'low R for matrix, it is not inversibility'
        exit()
        return
    
##  Step:5 final_function for w = [b, w1, w2, w3, ..., wn] 
    w = np.dot( np.dot( np.linalg.inv( Func_1 ) , X ) ,  label ) 
    
    return w


    
def predict( data, w ):
    
    ## 判断数据的特征维数 和 w 的维数是不是相同
    if data.shape[0] + 1 != w.shape[0]:
        print 'the row of data should be equal to the row w', w.shape[0] - 1
        exit()
    ## 构造一个全 1的 行向量 其中 列 等于  矩阵的列的维度     
    First_line = np.array(np.ones(data.shape[1]))
    
    # 加到 X 矩阵的第一行
    X = np.vstack(( First_line , data ))
    
    ## 预测过程
    Y = np.dot( X.transpose() , w )
    
    return Y


## test set            
if __name__=='__main__':
    ################
   train_data = np.array([[2,3,2,5],[0,1,2,1]])
   # print train_data
   label = [1,23,3,4]
   solution = train(train_data, label )
   print 'my method corref_', solution   
   judge = predict( train_data, solution) 
   print 'my_judge',judge
   #################
   clf = linear_model.LinearRegression()
   clf.fit(np.transpose(train_data),label)
   print 'official method corref_',clf.coef_
   #print 'official_judge',judge 
   predict = clf.decision_function(np.transpose(train_data))