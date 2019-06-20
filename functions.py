#! /usr/bin/env python
#coding=utf-8
#!/usr/bin/env python
from math import *
import math
#import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#from mpl_toolkits.basemap import Basemap 
from sklearn.cluster import DBSCAN
#import sets
from random import randrange  
from sklearn.datasets import make_blobs  
from sklearn.preprocessing import normalize  

from sklearn.preprocessing import LabelBinarizer
#import mnist_loader
#import network

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
#import seaborn as sns
####################################################################################################### 
def distanceNorm(Norm,D_value):  
    # initialization     
    # Norm for distance  
    if Norm == '1':  
        counter = np.absolute(D_value);  
        counter = np.sum(counter);  
    elif Norm == '2':  
        counter = np.power(D_value,2);  
        counter = np.sum(counter);  
        counter = np.sqrt(counter);  
    elif Norm == 'Infinity':  
        counter = np.absolute(D_value);  
        counter = np.max(counter);  
    else:  
        raise Exception('We will program this later......');  
  
    return counter;  
  
  
  
def fit(features,labels,iter_ratio,k,norm):  
    # initialization  
    (n_samples,n_features) = np.shape(features);  
    distance = np.zeros((n_samples,n_samples));  
    weight = np.zeros(n_features);  
    labels = map(int,labels)  
  
    # compute distance  
    for index_i in xrange(n_samples):  
        for index_j in xrange(index_i+1,n_samples):  
            D_value = features[index_i] - features[index_j];  
            distance[index_i,index_j] = distanceNorm(norm,D_value);  
    distance += distance.T;  
      
  
    # start iteration  
    for iter_num in xrange(int(iter_ratio*n_samples)):  
        # random extract a sample  
        index_i = randrange(0,n_samples,1);  
        self_features = features[index_i];  
  
        # initialization  
        nearHit = list();  
        nearMiss = dict();  
        n_labels = list(set(labels));  
        termination = np.zeros(len(n_labels));  
        del n_labels[n_labels.index(labels[index_i])];  
        for label in n_labels:  
            nearMiss[label] = list();  
        distance_sort = list();  
  
          
        # search for nearHit and nearMiss  
        distance[index_i,index_i] = np.max(distance[index_i]);      # filter self-distance   
        for index in xrange(n_samples):  
            distance_sort.append([distance[index_i,index],index,labels[index]]);  
  
        distance_sort.sort(key = lambda x:x[0]);  
  
        for index in xrange(n_samples):  
            # search nearHit  
            if distance_sort[index][2] == labels[index_i]:  
                if len(nearHit) < k:  
                    nearHit.append(features[distance_sort[index][1]]);  
                else:  
                    termination[distance_sort[index][2]] = 1;  
            # search nearMiss  
            elif distance_sort[index][2] != labels[index_i]:  
                if len(nearMiss[distance_sort[index][2]]) < k:  
                    nearMiss[distance_sort[index][2]].append(features[distance_sort[index][1]]);  
                else:  
                    termination[distance_sort[index][2]] = 1;  
  
            if map(int,list(termination)).count(0) == 0:  
                break;  
  
        # update weight  
        nearHit_term = np.zeros(n_features);  
        for x in nearHit:  
            nearHit += np.abs(np.power(self_features - x,2));  
        nearMiss_term = np.zeros((len(list(set(labels))),n_features));  
        for index,label in enumerate(nearMiss.keys()):  
            for x in nearMiss[label]:  
                nearMiss_term[index] += np.abs(np.power(self_features - x,2));  
            weight += nearMiss_term[index]/(k*len(nearMiss.keys()));  
        weight -= nearHit_term/k;  
          
  
    # print weight/(iter_ratio*n_samples);  
    return weight/(iter_ratio*n_samples);  

def readfile_as_df(filelist):
    files = './datafile/'
    mergedf = pd.DataFrame()
    mergedf = pd.read_csv(files+'ragender.csv',index_col=[0])
    for i in filelist:
        newdf = pd.read_csv(files+i,index_col=[0])
        if i in ['r4depres.csv', 'r4sleepr.csv','r4whappy.csv','r4flone.csv','r4fsad.csv','r4going.csv','r4enlife.csv','r4effort.csv',
                 'r5depres.csv', 'r5sleepr.csv','r5whappy.csv','r5flone.csv','r5fsad.csv','r5going.csv','r5enlife.csv','r5effort.csv',
                 'r6depres.csv', 'r6sleepr.csv','r6whappy.csv','r6flone.csv','r6fsad.csv','r6going.csv','r6enlife.csv','r6effort.csv',
                 'r7depres.csv', 'r7sleepr.csv','r7whappy.csv','r7flone.csv','r7fsad.csv','r7going.csv','r7enlife.csv','r7effort.csv',
                 'r8depres.csv', 'r8sleepr.csv','r8whappy.csv','r8flone.csv','r8fsad.csv','r8going.csv','r8enlife.csv','r8effort.csv',
                 'r9depres.csv', 'r9sleepr.csv','r9whappy.csv','r9flone.csv','r9fsad.csv','r9going.csv','r9enlife.csv','r9effort.csv',
                 'r10depres.csv', 'r10sleepr.csv','r10whappy.csv','r10flone.csv','r10fsad.csv','r10going.csv','r10enlife.csv','r10effort.csv',
                 'r11depres.csv', 'r11sleepr.csv','r11whappy.csv','r11flone.csv','r11fsad.csv','r11going.csv','r11enlife.csv','r11effort.csv',
                 'r12depres.csv', 'r12sleepr.csv','r12whappy.csv','r12flone.csv','r12fsad.csv','r12going.csv','r12enlife.csv','r12effort.csv',
                 'r4sayret.csv', 'r5sayret.csv','r6sayret.csv','r7sayret.csv','r8sayret.csv','r9sayret.csv', 'r10sayret.csv','r11sayret.csv','r12sayret.csv']:
            #newdf = newdf.fillna(value = '3.vague')
            'nothing'
        mergedf = pd.merge(mergedf, newdf, on='hhidpn')
    return mergedf

def reformat(mergedf, startwave, endwave):    
    cleaned = mergedf.dropna() #这里，dropnan的方法
    cleaned.index = range(len(cleaned.index)) #reindex 注意总结，这里drop以后，更新index
                    
    ##change data type
    if 'ragender' in cleaned.columns:
        temp=pd.Series(int(i[:1]) for i in cleaned['ragender'])
        cleaned['ragender']=temp
        
    if 'raedegrm' in cleaned.columns:        
        temp=pd.Series(int(i[:1]) for i in cleaned['raedegrm'])
        cleaned['raedegrm']=temp

    for k in range(startwave,endwave+1):
        if 'r{0}cancre'.format(k) in cleaned.columns:
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}cancre'.format(k)])
            cleaned['r{0}cancre'.format(k)]=temp[temp<1.1]
        
        if 'r{0}sayret'.format(k) in cleaned.columns:       
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}sayret'.format(k)])
            cleaned['r{0}sayret'.format(k)]=temp[temp<3.1]
        
        if 'r{0}diab'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}diab'.format(k)])
            cleaned['r{0}diab'.format(k)]=temp[temp<1.1]
        
        if 'r{0}depres'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}depres'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}depres'.format(k)]=temp
        
        if 'r{0}sleepr'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}sleepr'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}sleepr'.format(k)]=temp
        
        if 'r{0}whappy'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}whappy'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}whappy'.format(k)]=temp
            
        if 'r{0}flone'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}flone'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}flone'.format(k)]=temp
            
        if 'r{0}fsad'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}fsad'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}fsad'.format(k)]=temp
            
        if 'r{0}going'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}going'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}going'.format(k)]=temp
            
        if 'r{0}enlife'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}enlife'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}enlife'.format(k)]=temp
            
        if 'r{0}effort'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}effort'.format(k)])
            #temp[temp == 2]=0.5
            cleaned['r{0}effort'.format(k)]=temp
            
    cleaned = cleaned.dropna()
    cleaned.index = range(len(cleaned.index)) #reindex 注意总结，这里drop以后，更新index  
    tempw = deepcopy(cleaned)        
    for k in range(startwave,endwave+1):
        if 'r{0}drinkd'.format(k) in cleaned.columns:  
            temp=pd.Series(int(float(i[:3])) for i in cleaned['r{0}drinkd'.format(k)])   #zongjie: in fact, a operation about series
            cleaned['r{0}drinkd'.format(k)]=temp
        
        if 'r{0}vgactx'.format(k) in cleaned.columns:
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}vgactx'.format(k)])
            cleaned['r{0}vgactx'.format(k)]=temp
        
        if 'r{0}iearn'.format(k) in cleaned.columns:       
            temp=pd.Series(int(i) for i in cleaned['r{0}iearn'.format(k)])
            cleaned['r{0}iearn'.format(k)]=temp
        
        if 'h{0}hhres'.format(k) in cleaned.columns:       
            temp=pd.Series(int(i) for i in cleaned['h{0}hhres'.format(k)])
            cleaned['h{0}hhres'.format(k)]=temp   
        
        if 'r{0}shltc'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i) for i in cleaned['r{0}shltc'.format(k)])
            cleaned['r{0}shltc'.format(k)]=temp
        
        if 's{0}hhidpn'.format(k) in cleaned.columns:
            temp2=deepcopy(tempw)
            temp2[temp2['s{0}hhidpn'.format(k)]>0]=1
            cleaned['s{0}hhidpn'.format(k)]=temp2['s{0}hhidpn'.format(k)]
        
        if 'r{0}mstat'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}mstat'.format(k)])
            cleaned['r{0}mstat'.format(k)]=temp
        
        if 'r{0}work'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}work'.format(k)])
            cleaned['r{0}work'.format(k)]=temp
                    
        if 'h{0}child'.format(k) in cleaned.columns:  
            temp2=deepcopy(tempw)
            temp2[temp2['h{0}child'.format(k)]>0]=1
            cleaned['h{0}child'.format(k)]=temp2['h{0}child'.format(k)]
        
        if 'r{0}hosp'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}hosp'.format(k)])
            cleaned['r{0}hosp'.format(k)]=temp 
                    
        if 'r{0}walkr'.format(k) in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned['r{0}walkr'.format(k)])
            cleaned['r{0}walkr'.format(k)]=temp 
        
        variname = 'r{0}smoken'.format(k)         
        if variname in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            cleaned[variname]=temp

        variname = 'r{0}shlt'.format(k)         
        if variname in cleaned.columns:        
            temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            cleaned[variname]=temp 
        
        variname = 'r{0}hsptim'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp
    
        variname = 'r{0}hspnit'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp

        variname = 's{0}hsptim'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp
            
        variname = 's{0}hspnit'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp

        variname = 's{0}nrsnit'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp

        variname = 'r{0}nrsnit'.format(k)         
        if variname in cleaned.columns:        
            if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
            else:temp=pd.Series(int(i) for i in cleaned[variname])
            cleaned[variname]=temp
        
        variname = 'rabyear'         
        if variname in cleaned.columns:        
            temp=pd.Series(2014-int(i) for i in cleaned[variname])
            cleaned[variname]=temp
            
        for n in ['r{0}doctor','r{0}doctim','r{0}drugs','r{0}outpt','r{0}dentst','r{0}spcfac','r{0}oopmd','r{0}hlthlm','r{0}phone','r{0}meds','r{0}money','r{0}shop','r{0}meals',
                  'r{0}map','r{0}calc','r{0}mcwv','r{0}comp','r{0}walksa','r{0}jog','r{0}climsa','r{0}clim1a','r{0}adla','r{0}adlwa','r{0}mobila','r{0}lamusa','r{0}grossa',   
                   'r{0}finea']:
            variname = n.format(k) 
            if variname in cleaned.columns: 
                if type(cleaned[variname][0]) is str:temp=pd.Series(int(i[:1]) for i in cleaned[variname])
                else:temp=pd.Series(int(i) for i in cleaned[variname])
                cleaned[variname]=temp
    
    return cleaned







