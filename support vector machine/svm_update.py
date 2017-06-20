# -*- coding: utf-8 -*-
## function : A linear classification for SVM 
## OOP based function
import matplotlib.pyplot as plt
import numpy as np
import random

class Support_Vector_Machine():
    
    # -------------------------------------------------------------------------
    def __init__(self ,  data , label , C , max_iteration ):  
        
      self.stop_tag = False  
      self.data =  data
      self.label = label
      self.C = C
      self.max_iteration = max_iteration
      self.alpha = np.zeros(data.shape[0])
      self.b = 0 
      self.alpha_1_index = None
      self.alpha_2_index = None
      self.E = np.zeros( data.shape[0] )
      self.w = None
    
    # --plot the line with w and b-----------------------------------------------------------
    def plot_line( self ):
     
      plt.figure()
      x = np.linspace(0, 10, 1000)  
    
      w_2 = (-1)*float( self.w[1]) / ( float( self.w[0]) + 0.0001 )
      print 'w_2 =',w_2
      b_2 = (-1)*self.b / float(self.w[1])
      print 'b_2 =',b_2
    
      y = w_2 * x + b_2
      #print 'y', y
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
    # -------------------------------------------------------------------------
    def param_w( self ):    
      w = 0
    
      for index in range(data.shape[0]):
          w = w + label[index] * data[index] * self.alpha[index]
          
      self.w = w
      
      print'w',w
      return
    
    # only for linear kernel
    def kernel( self , vec_1 , vec_2 ):  
      sum = 0     
      for index in range(vec_1.shape[0]):
        sum = sum + vec_1[index] * vec_2[index]          
      return sum
    
    # -------------------------------------------------------------------------
    # Calculate the function G
    # g(x) = sum( ai * yi * K(xi,x) ) + b
    def function_G( self , certain_index ):
        sum = 0     
        for index in range(data.shape[0]):
            sum = sum + self.alpha[index] * label[index] * self.kernel(data[index] , data[certain_index])   
            
        sum = sum + self.b   
        
        return sum  
    
    # -------------------------------------------------------------------------
    # Calculate the function E
    # Ei = g(xi) - yi
    def function_E( self , certain_index ):
        E_1 = self.function_G( certain_index )       
        E = E_1 - self.label[certain_index]      
        
        return E           
       
    # ------update E value-----------------------------------------------------
    def E_value_update( self ):
        for index in range( data.shape[0] ):
            self.E[index] = self.function_E( index )
        
        print 'E_value',self.E
        return
    
    # ------ update alpha series ----------------------------------------------
    def alpha_and_b_update( self ):
        # Calculate yita = K11 + K22 - 2 * K12 
        yita = self.kernel( data[self.alpha_1_index] , data[self.alpha_1_index] ) +  self.kernel( data[self.alpha_2_index] ,data[self.alpha_2_index]) - 2 *  self.kernel( data[self.alpha_1_index] , data[self.alpha_2_index] )
        
        #print 'alpha_1_index',self.alpha_1_index
        #print 'alpha_2_index',self.alpha_2_index
        
        alpha_2_new_unc = self.alpha[self.alpha_2_index] + self.label[self.alpha_2_index] \
                         * ( self.E[ self.alpha_1_index ] - self.E[ self.alpha_2_index ] ) / yita
                         
        print 'alpha-2-new-unc',alpha_2_new_unc
        # --------------------- P126 最下方 ----------------------------------------------------------------
        if self.label[self.alpha_1_index] == self.label[self.alpha_2_index]:
            L = max( 0 , self.alpha[self.alpha_1_index] + self.alpha[self.alpha_2_index] - self.C )
            H = min( self.C , self.alpha[self.alpha_2_index] + self.alpha[self.alpha_1_index])
            
        if self.label[self.alpha_1_index] != self.label[self.alpha_2_index]:
            L = max( 0 , self.alpha[self.alpha_2_index] - self.alpha[self.alpha_1_index] )
            H = min( self.C , self.C + self.alpha[self.alpha_2_index] - self.alpha[self.alpha_1_index])
        
        #--------------- 7.108 -----------------
        # ------------  update the alpha 2 value ------------------------------
        if alpha_2_new_unc > H :
            alpha_2_new = H
        if alpha_2_new_unc < L :
            alpha_2_new = L
        if alpha_2_new_unc >= L and alpha_2_new_unc <= H :
            alpha_2_new = alpha_2_new_unc
        
        # ------------ update the alpha 1 value -------------------------------
        alpha_1_new = self.alpha[self.alpha_1_index] + self.label[self.alpha_1_index] * self.label[self.alpha_2_index] * ( self.alpha[self.alpha_2_index] - alpha_2_new )
        
        self.alpha[self.alpha_1_index]  = alpha_1_new
        self.alpha[self.alpha_2_index]  = alpha_2_new
        
        print 'alpha-1-new index',self.alpha_1_index,'alpha-2-new index',self.alpha_2_index
        print 'alpha-1-new',alpha_1_new,'alpha-2-new',alpha_2_new
        print 'alpha-update',self.alpha
        
        b1_new = - self.function_E( self.alpha_1_index ) \
                 - label[self.alpha_1_index] * self.kernel( data[self.alpha_1_index] , data[self.alpha_1_index]) \
                 * ( alpha_1_new - self.alpha[self.alpha_1_index] ) \
                 - label[self.alpha_2_index] * self.kernel( data[self.alpha_2_index] , data[self.alpha_1_index]) \
                 * ( alpha_2_new - self.alpha[self.alpha_2_index] ) \
                 + self.b
     
        b2_new = - self.function_E( self.alpha_2_index ) \
                 - label[self.alpha_1_index] * self.kernel( data[self.alpha_1_index] , data[self.alpha_2_index]) \
                 * ( alpha_1_new - self.alpha[self.alpha_1_index] ) \
                 - label[self.alpha_2_index] * self.kernel( data[self.alpha_2_index] , data[self.alpha_2_index]) \
                 * ( alpha_2_new - self.alpha[self.alpha_2_index] ) \
                 + self.b
                 
        print ' b1_new ',b1_new,' b2_new ',b2_new
        b_new = ( b1_new + b2_new ) / 2
        
        self.b = b_new
        
        print 'b-update',b_new
        
        return
        
    # -------------------------------------------------------------------------
    # 寻找最优的 alpha_1
    # 选择 alpha-1 alpha-2  
    # 1) 0 < alpha-1 < C  选 |E1 - E2|最大的
    # 2) 其他情形  选 alpha-1 = 0   or  alpha-1 = C
    def search_best_alpha_1( self ):
        support_vector = []
        point_other = []
        # --------------------------- 7.111 ~ 7.113 ---------------------------
        for index in range(self.alpha.shape[0]):
            tmp_num = self.alpha[index]
            _JUDGE_ = self.label[index] * self.function_G(index) 
            if tmp_num > 0 and tmp_num < C and _JUDGE_ != 1:
                support_vector.append(index)
            if ( tmp_num == 0 and _JUDGE_ < 1 ) or ( tmp_num == C and _JUDGE_ > 1 ) :
                point_other.append(index)
        
        support_vector = np.array( support_vector )
        point_other = np.array( point_other )
        
        print 'support_vector',support_vector
        print 'point_other',point_other
        
        # ------------------------优先选择支持向量------------------------------
        if support_vector.shape[0] != 0 :
           alpha_1_index = random.choice(support_vector)
           self.alpha_1_index = alpha_1_index
        
        if support_vector.shape[0] == 0 and point_other.shape[0] != 0 :
            alpha_1_index = random.choice(point_other)
            self.alpha_1_index = alpha_1_index
       
        if support_vector.shape[0] == 0 and point_other.shape[0] == 0 :
            self.stop_tag = True
            self.alpha_1_index = None
        
        #print 'alpha-1-index',self.alpha_1_index
        return
        # ---------------------------------------------------------------------
    
    # 寻找最优的 alpha_2
    #  alpha_1 已经确定 E1 确定 只需找 alpha_2 最大就行
    #  根据 |E1 - E2| max 选取 最大的
    def search_best_alpha_2( self ):
        #self.E_value_update()
        E1 = self.E[self.alpha_1_index]
        E = []
        #print 'E_series',self.E
        # ----------------- Calculate the E value ---------------------------
        for index in range( data.shape[0] ):
            E.append( E1 - self.E[index] ) 
            
        # --------------- Search the largest E value ------------------------
        print 'original_E',E
        index = np.argsort( abs(np.array(E)) )
        
       
        print 'index',index
        
        alpha_2_index = index[ index.shape[0] - 1 ]
        alpha_2_index_2 = index[ index.shape[0] - 2 ]
        
        if alpha_2_index != self.alpha_1_index:
           self.alpha_2_index = alpha_2_index
           
        if alpha_2_index == self.alpha_1_index:
           self.alpha_2_index = alpha_2_index_2
        
        return 
      # -----------------------------------------------------------------------
    # --------------- train_process for Support Vector Machine ---------------- 
    def train( self ):
       Iter = 0  
       #print 'Iter',Iter
       while  ( self.stop_tag == False ) and ( Iter < self.max_iteration ) :
          print ' '
          print 'E-value-update'
          self.E_value_update()
          
          print 'search-best-alpha-1'
          self.search_best_alpha_1()
          
          print 'search-best-alpha-2'
          self.search_best_alpha_2()
          
          print 'update alpha-and-b'
          self.alpha_and_b_update()
          
          print 'calculate the param w'
          self.param_w()
          
          print 'plot line'
          self.plot_line()
          
          Iter = Iter + 1
          print Iter
        
      
if __name__=='__main__':
    #data = np.array([[1,2],[2,4],[3,3],[6,6],[8,5],[9,9],[10,14]])
    #label = np.array([1, 1 , 1 ,-1,-1,-1,-1])
    #C = 0.2
    data = np.array([[1,2],[2,4],[3,3],[6,6],[8,5],[9,9],[10,14]])
    label = np.array([1, 1 , 1 ,-1,-1,-1,-1])
    C = 0.002
    Max_iterations = 200
    #-------------------主函数测试----------------------------------------
    A = Support_Vector_Machine( data , label , C , Max_iterations)
    
    #--------------function_G 函数测试---------------------------------
    C_1 = A.function_G( 1 )
    print C_1
    
    #---------------function train -------------------------------------
    A.train()