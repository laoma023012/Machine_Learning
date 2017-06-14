# -*- coding: utf-8 -*-


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
