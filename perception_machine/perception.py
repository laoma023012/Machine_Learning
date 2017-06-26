# -*- coding: utf-8 -*-
import numpy as np

class perception():
    def __init__( self , data , label , learning_rates , max_iterations):
        self.data = data
        self.label = label
        self.w = np.zeros(( 1 , data.shape[1] ))
        self.b = 0
        self.learning_rate = learning_rates
        self.continue_tag = True
        self.max_iterations = max_iterations
        
    def train( self ):
        
        Tmp_right = []
        
        count = 0
        
        while self.continue_tag == True or count <= self.max_iterations:
          for index in range( self.data.shape[0] ):
            # update w and b if y * ( w*x + b) <= 0
            JUDGE = np.dot( self.label[index] , np.dot(self.w, self.data[index].transpose()) )
            #print ('JUDGE',JUDGE)
            #print ('w',self.w)
            if JUDGE <= 0:
                # w = w + learning_rate * label[index] * data[index].T
                self.w = self.w + self.learning_rate * self.label[index] * self.data[index].transpose()
                #print (self.w)
                # b = b + learning_rate * label[index]
                self.b = self.b + self.learning_rate * self.label[index] * self.data[index].transpose()
            if JUDGE > 0:
                Tmp_right.append(index)
                if self.data.shape[0] == len(set(Tmp_right)):
                    self.continue_tag = False
               
            count = count + 1
            
        return Tmp_right
        
    def predict( self , data ):
         predict = []
         #print (data)
         for index in range(data.shape[0]):
             
             tmp_predict = np.dot( self.w , data[index].transpose() )
             print ('tmp_predict',tmp_predict)
             tmp_predict = np.sign( tmp_predict )
             predict.append( tmp_predict )

if __name__ == '__main__':
     
    data = np.array([[1,5],[5,3],[6,3]])
    label = np.array([1,-1,1])
    learning_rate = 0.1
    max_iterations = 100
    A = perception( data , label , learning_rate , max_iterations )
    A.train()
    predict_label = A.predict( data )
    print (predict_label)           
                
                
