# -*- coding: utf-8 -*-
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# function : search best alpha_1
#             first loop for 0 < alpha < C
#             second loop for alpha_i = C or alpha_i = 0 
def plot_pic( data , label , omiga , b):
    data = np.array(data)
    plt.figure()
    x = np.linspace(0, 10, 1000)  
    #y = w * x + b
    #plt.plot(x,y)
    w_2 = (-1)*float(omiga[1]) / float(omiga[0])
    print 'w_2 =',w_2
    b_2 = (-1)*b/ float(omiga[1])
    print 'b_2 =',b_2
    
    y = w_2 * x + b_2
    print 'y', y
    plt.plot(x,y)
    data_1 = []
    data_2 = []
    
    for index in range(data.shape[0]):
        if label[index] == 1:
            data_1.append(data[index])
        if label[index] == -1:
            data_2.append(data[index])
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    plt.scatter(data_1[:,0],data_1[:,1])
    plt.scatter(data_2[:,0],data_2[:,1])
    return

def param_w( data , label , alpha ,b ):
    w = np.zeros( data.shape[1] )
    print w
   
    for index in range( data.shape[0] ):
        print alpha[index]
        #prin
        w = w + alpha[index] * label[index] * data[index]
    return w

def update_alpha_b( alpha , alpha_1_index , alpha_2_index , data , label , E1 , E2 , b , C ):
#----------------- update alpha_series-------------------------- 
    # calculate L H
   if label[alpha_1_index] != label[alpha_2_index]:
       L = max( 0 , alpha[alpha_2_index] - alpha[alpha_1_index] )
       H = min( C , C + alpha[alpha_2_index] - alpha[alpha_1_index] )
   if label[alpha_1_index] == label[alpha_2_index]: 
       L = max( 0 , alpha[alpha_2_index] + alpha[alpha_1_index] - C )
       H = min( C , alpha[alpha_2_index] + alpha[alpha_1_index] )
       
   yita = kernel(data[alpha_1_index],data[alpha_1_index]) + kernel(data[alpha_2_index],data[alpha_2_index]) - 2 * kernel(data[alpha_1_index],data[alpha_2_index])
   alpha_2_new_unc = alpha[alpha_2_index] + label[alpha_2_index] * ( E1 - E2 ) / yita
   
   if alpha_2_new_unc > H:
       alpha_2_new = H
   if alpha_2_new_unc < L:
       alpha_2_new = L
   if alpha_2_new_unc >= L and alpha_2_new_unc <= H:
       alpha_2_new = alpha_2_new_unc
       
   alpha_1_new = alpha[alpha_1_index] + label[alpha_1_index] * label[alpha_2_index] * ( alpha[alpha_2_index] - alpha_2_new )
   print 'alpha_2_new',alpha_2_new,'alpha_1_new',alpha_1_new
   #print 'alpha_1',alpha[alpha_1_index]
   alpha_new = copy.deepcopy(alpha)
   #print alpha_new
   alpha_new[alpha_1_index] = alpha_1_new
   alpha_new[alpha_2_index] = alpha_2_new

#------------------------------ b1 new and b2 new ----------------------------------
   #print 'KERNEL11',kernel(data[alpha_1_index],data[alpha_1_index])
   #print 'KERNEL21',kernel(data[alpha_2_index],data[alpha_1_index])
   #print 'KERNEL12',kernel(data[alpha_1_index],data[alpha_2_index])
   #print 'KERNEL22',kernel(data[alpha_2_index],data[alpha_2_index])
   #print 'Y1',label[alpha_1_index]
   #print 'Y2',label[alpha_2_index]
   #print 'ans_1',label[alpha_1_index] * kernel(data[alpha_1_index],data[alpha_1_index]) 
   #print 'ans_2',alpha
   b1_new = (-1) * E1 - label[alpha_1_index] * kernel(data[alpha_1_index],data[alpha_1_index]) * ( alpha_1_new - alpha[alpha_1_index] ) - label[alpha_2_index] * kernel(data[alpha_2_index],data[alpha_1_index]) * ( alpha_2_new - alpha[alpha_2_index] ) + b
   b2_new = -E2 - label[alpha_1_index] * kernel(data[alpha_1_index],data[alpha_2_index]) * ( alpha_1_new - alpha[alpha_1_index] ) - label[alpha_2_index] * kernel(data[alpha_2_index],data[alpha_2_index]) * ( alpha_2_new - alpha[alpha_2_index] ) + b
   b_new = (b1_new + b2_new) / 2
   #print alpha_1_new , alpha_2_new
   print 'b1_new', b1_new , 'b2_new' , b2_new
   return alpha_new  , b_new

def _search_best_alpha_1_( alpha , b , data, label, C ):
   # 先取支持向量 
   alpha_1_index_important = []
   for index in range(alpha.shape[0]):
        if alpha[index] > 0 and alpha[index] < C:
            JUDGE = label[index] * function_G( alpha , data , label , index , b)
            if int(JUDGE) != 1:
                #print type(JUDGE)
                #print 'JUDGEHAHA',JUDGE
                #print 'index',index
                alpha_1_index_important.append(index)
                #return index
   
   # 如果支持向量都满足 则
   sublist = []
   for index in range(alpha.shape[0]):
        if alpha[index] == 0:  
             JUDGE = label[index] * function_G( alpha , data , label , index , b )
             if JUDGE < 1:
                 sublist.append(index)
        if alpha[index] == C:
            JUDGE = label[index] * function_G( alpha , data , label , index , b )
            if JUDGE > 1:
                 sublist.append(index)
   #print 'sublist',sublist
   #print 'size of sublist',np.array(sublist).shape[0]
   if np.array(alpha_1_index_important).shape[0] != 0:
       alpha_1_index = random.choice(alpha_1_index_important)
       #print 'alpha_1_index_size',np.array(alpha_1_index_important).shape[0]
       return alpha_1_index
   
   if np.array(alpha_1_index_important).shape[0] == 0 and np.array(sublist).shape[0] != 0:
       alpha_1_index = random.choice(sublist)
       #print 'sublist',np.array(sublist).shape[0]
       return alpha_1_index
   
   if np.array(sublist).shape[0] == 0:
       print 'correct'
       exit(0)
       return
   

def _search_best_alpha_2( alpha , b , alpha_1_index, data, label, C ):
    E1 = function_E( alpha ,data , label , alpha_1_index , b)
    #print E1
    E2 = []
    
    for index in range(alpha.shape[0]):
       E_value = function_E( alpha , data , label , index , b) 
       E2.append(E_value)
    
    print 'E2',E2
    E = abs(E1 - E2)
    index = np.argsort(E)
    
    best_alpha_2_index = index[index.shape[0] - 1]
    return best_alpha_2_index , E1 , E2[best_alpha_2_index]

def function_E( alpha , data , label , certain_figure_index , b):
    E_1 = function_G( alpha, data , label , certain_figure_index , b )
    #print E_1
    E = E_1 - label[certain_figure_index]
    return E

def function_G( alpha, data , label , certain_figure_index , b):
    
    sum = 0
    
    for index in range(data.shape[0]):
        sum = sum + alpha[index] * label[index] * kernel(data[index] , data[certain_figure_index])
    
    sum = sum + b
    return sum

def kernel( vec_1 , vec_2 ):
   
    sum = 0
    
    for index in range(vec_1.shape[0]):
        #print 'index',index
        #print 'vec_1[index]',vec_1[index]
        sum = sum + vec_1[index] * vec_2[index]    
    
    return sum

    
# function : init alpha and b
#            select the first choice alpha randomly
    
def _init_alpha_b_( data ):
   init_alpha = np.zeros(data.shape[0]) 
   b = 0
   first_pick_alpha_1 = random.randint(0,data.shape[0] - 1)
   #print first_pick_alpha_1 
   return init_alpha, b , first_pick_alpha_1
    
def train_svm( data , label , C ):
    # first_round for alpha-1 and alpha-2 pick
    alpha , b , alpha_1_index = _init_alpha_b_( data )
    alpha_2_index , E1 , E2 = _search_best_alpha_2( alpha , b , alpha_1_index , data , label , C )
    alpha , b = update_alpha_b( alpha , alpha_1_index , alpha_2_index , data , label , E1 , E2 , b , C )
    print 'alpha_1_new', alpha
    print 'b_1_new', b
    
    print ' '
    ## second round for alpha-1 and alpha-2 pick
    for index in range(20):
      alpha_1_index = _search_best_alpha_1_( alpha , b , data, label, C )
      alpha_2_index , E1 , E2 = _search_best_alpha_2( alpha , b , alpha_1_index , data , label , C )
      alpha , b = update_alpha_b( alpha , alpha_1_index , alpha_2_index , data , label , E1 , E2 , b , C )
      w = param_w( data , label , alpha ,b )
      plot_pic( data , label , w , b)
      print 'alpha_2_new', alpha
      print 'w',w
      print 'b_2_new', b
    #-----------------------------------------------------
    
    

if __name__=='__main__':
    data = np.array([[1,2],[2,4],[3,3],[6,6],[8,5],[9,9],[10,14]])
    label = np.array([1, 1 , 1 ,-1,-1,-1,-1])
    C = 0.002
    train_svm(data , label , C )