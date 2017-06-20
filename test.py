# -*- coding: utf-8 -*-
import numpy as np

class test():
     def __init__(self ):  
         self.alpha = np.zeros(10)
     def show(self):
         self.alpha[0] = 1
     def main(self):
         self.show()
         for index in range( self.alpha.shape[0] ):
             print self.alpha[index]
             
if __name__=='__main__':
    A = test()
    A.main()