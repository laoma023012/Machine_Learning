# -*- coding: utf-8 -*-
import math
import operator

class decision_tree():
    def __init__(self , data , label):
        self.data =  data
        self.label =  label
    
    def spiltDataSet( self , axis ,value):
        retDataSet = []
        for featVec in self.data:
            if featVec[axis] == value:
                # chop out axis used for spiltting
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        
        return retDataSet
    
    def calcShannonEnt( self ):
        numEntries = len( self.data )
        
        labelCounts = {}
        
        for featVec in self.data:
            currentLabel = featVec[-1]
            
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            
            labelCounts[currentLabel] += 1
        
        shannonEnt = 0.0

        for key in labelCounts:
            
            prob = float( labelCounts[key] ) / numEntries
            shannonEnt = shannonEnt - prob * math.log( prob , 2 )
            
        return shannonEnt
    
    def chooseBestFeatureToSplit( self ):
        # the last column is used for the labels
        numFeatures = len(self.data[0]) - 1
        baseEntropy = self.calcShannonEnt( self.data )
        bestInfoGain = 0.0
        bestFeature = -1
        # iterate over all the features
        for i in range( numFeatures ):
            # create list for all the examples of this feature
            featList = [example[i] for example in self.data]
            # get a set of unique valus
            uniqueVals = set(featList)
            newEntropy = 0.0
            
            for value in uniqueVals:
                subDataSet = self.spiltDataSet( i , value)
                prob = len(subDataSet) / float(len(self.data))
                newEntropy = newEntropy + prob * self.calcShannonEnt( self.data )
            
            # calculate the info gain
            infoGain = baseEntropy - newEntropy
            # compare this to the best gain so far
            if ( infoGain > bestInfoGain ):
                #if better than current best set to best
                bestInfoGain = infoGain
                bestFeature = i
            
        return bestFeature
    
    def majorityCnt(self , classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():classCount[vote] = 0
            classCount[vote] += 1
                      
        sortedClassCount = sorted(classCount.iteritems() , key = operator.itemgetter(1) , reverse = True)
        return sortedClassCount[0][0]                               
    
    def createTree(self):
        classList = [example[-1] for example in self.data]
        
        # stop spilting when all of the classes are equal
        if classList.count(classList[0] == len(classList)):
            return classList[0]
        
        # stop spilting when there are no more freatures in dataset
        if len(self.data) == 1:
            return self.majorityCnt(classList)
        
        bestFeat = self.chooseBestFeatureToSplit( self.data )
        bestFeatLabel = self.label[bestFeat]
        myTree = { bestFeatLabel:{} }
        del( self.label[bestFeatLabel] )
        featValues = [ example[bestFeat] for example in self.data ]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # copy all of labels , so trees don't mess up existing labels
            subLabels = self.label[:]
            
            myTree[ bestFeatLabel ][value] = self.createTree( self.spiltDataSet(bestFeat,value), subLabels )
            
        return myTree
        
    def classify( self , inputTree , featLabels , testVec ):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        
        if isinstance(valueOfFeat , dict):
            classLabel = self.classify(valueOfFeat , featLabels , testVec)
        else:
            classLabel = valueOfFeat
        return classLabel
    
    def getResult():
        