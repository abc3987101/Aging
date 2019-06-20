#! /usr/bin/env python
#coding=utf-8
#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from functions import readfile_as_df
from functions import reformat
#%%

#==========CONSTANT NUMBER=======#

#adjust to perform experiment 
#on balanced/unbalanced test data
balance_test = True
random_seed = 3

#first cohort 
sw=5  
#terminal cohort
ew=11 

#adjust this number to perform experiment 
#on different length of time window
varyLength = 6 

#==========READ TRAINING DATA=======#
fileslist=[]
for i in range(sw,ew+1):
    fileslist.append( 's{0}hhidpn.csv'.format(i) )
    fileslist.append( 'r{0}cesd.csv'.format(i) )
    fileslist.append( 'r{0}cancre.csv'.format(i) )
    fileslist.append( 'r{0}sayret.csv'.format(i) ) 
    fileslist.append( 'r{0}diab.csv'.format(i) )
    fileslist.append('r{0}smoken.csv'.format(i))  
    fileslist.append('r{0}shlt.csv'.format(i))  
    fileslist.append('r{0}mobila.csv'.format(i)) 
    fileslist.append('r{0}lgmusa.csv'.format(i)) 
#==data formation==
fileslist.append('rabyear.csv')
mergedf = readfile_as_df (fileslist)                           
cleaned = reformat(mergedf, startwave=sw, endwave=ew)
cleaned['nsp']=0
temp7=deepcopy(cleaned)
temp7[temp7['s{0}hhidpn'.format(ew)]==1]=1
temp7[temp7['s{0}hhidpn'.format(ew)]==0]=0
cleaned['nsp']=temp7['nsp']
cleaned['ragender'] = cleaned['ragender']-1
cleaned.index = range(len(cleaned.index))              
restore=deepcopy(cleaned)

#%% 
#====labels====       
lbsname='r{0}cesd'
lbs=lbsname.format(ew) 
cleaned['cesd_score'] = deepcopy(cleaned[lbs])          
temp = deepcopy(restore)
temp[temp[lbsname.format(ew)]<2.01]= int(0)  
temp[ temp[lbsname.format(ew)]>2.01]= int(1)  
cleaned[lbs] = temp[lbs] 
cleaned=cleaned[(cleaned[lbs]==0) | (cleaned[lbs]==1)] 

ai1=cleaned[cleaned[lbs]==1] 
ai2=cleaned[cleaned[lbs]==0] 
temp = ai2.sample(n=ai1.shape[0],random_state = random_seed) 

cleaned = ai1.append(temp)
cleaned.index = range(len(cleaned.index)) 

feature_names= []
for i in range(6,0,-1):
    feature_names= feature_names+['r{0}shlt'.format(ew-i),'r{0}mobila'.format(ew-i),'r{0}lgmusa'.format(ew-i),'r{0}smoken'.format(ew-i),
                                 'r{0}diab'.format(ew-i),'r{0}cancre'.format(ew-i),'r{0}cesd'.format(ew-i),'r{0}sayret'.format(ew-i),'nsp','ragender','rabyear']
cleaned['ragender'] = cleaned['ragender'].astype('category')
features = cleaned[feature_names] 
features = features.values
features.tolist()

labels=cleaned['cesd_score']

labels = [int(labels.ix[i]) for i in labels.index] 
features_nn=deepcopy(features)
labels_nn=deepcopy(labels)
features_nn = np.array(features_nn)

temp = int(len(feature_names)/(ew-sw+1-1))
features_nn_varyLength = features_nn[:,-1*temp*varyLength:]

min_max_scaler = preprocessing.MinMaxScaler((0,1))
features_nn_varyLength = min_max_scaler.fit_transform(features_nn_varyLength)
training_data_x, test_data_x, training_data_y ,test_data_y = train_test_split(features_nn_varyLength, labels_nn,test_size = 0.01,random_state=random_seed)

train_y1 = pd.Series(training_data_y)
train_y1[train_y1<=2]=0
train_y1[train_y1>2]=1

#%%
##========READ TEST DATA=========##
sw=sw+1
ew=ew+1
fileslist=[]
for i in range(sw,ew+1):
    fileslist.append( 's{0}hhidpn.csv'.format(i) )
    fileslist.append( 'r{0}cesd.csv'.format(i) )
    fileslist.append( 'r{0}cancre.csv'.format(i) )
    fileslist.append( 'r{0}sayret.csv'.format(i) ) 
    fileslist.append( 'r{0}diab.csv'.format(i) )
    fileslist.append('r{0}smoken.csv'.format(i))  
    fileslist.append('r{0}shlt.csv'.format(i))  
    fileslist.append('r{0}mobila.csv'.format(i)) 
    fileslist.append('r{0}lgmusa.csv'.format(i)) 
#===data formation
fileslist.append('rabyear.csv')
mergedf = readfile_as_df (fileslist)                           
cleaned = reformat(mergedf, startwave=sw, endwave=ew)
cleaned['nsp']=0
temp7=deepcopy(cleaned)
temp7[temp7['s{0}hhidpn'.format(ew)]==1]=1
temp7[temp7['s{0}hhidpn'.format(ew)]==0]=0
cleaned['nsp']=temp7['nsp']
cleaned['ragender'] = cleaned['ragender']-1
cleaned.index = range(len(cleaned.index))              
restore=deepcopy(cleaned)
#%% 
lbsname='r{0}cesd'
lbs=lbsname.format(ew) 
cleaned['cesd_score'] = deepcopy(cleaned[lbs])          
temp = deepcopy(restore)
temp[temp[lbsname.format(ew)]<2.01]= int(0)  
temp[ temp[lbsname.format(ew)]>2.01]= int(1)  
cleaned[lbs] = temp[lbs] 
cleaned=cleaned[(cleaned[lbs]==0) | (cleaned[lbs]==1)] 

ai1=cleaned[cleaned[lbs]==1] 
ai2=cleaned[cleaned[lbs]==0] 
if balance_test: temp = ai2.sample(n=ai1.shape[0] ,random_state=random_seed) 
else:            temp = ai2.sample(n=ai2.shape[0] ,random_state=random_seed) 
cleaned = ai1.append(temp)
cleaned.index = range(len(cleaned.index)) 
  
feature_names_testpart = []
for i in range(6,0,-1):
    feature_names_testpart= feature_names_testpart+['r{0}shlt'.format(ew-i),'r{0}mobila'.format(ew-i),'r{0}lgmusa'.format(ew-i),'r{0}smoken'.format(ew-i),
                                                  'r{0}diab'.format(ew-i),'r{0}cancre'.format(ew-i),'r{0}cesd'.format(ew-i),'r{0}sayret'.format(ew-i),'nsp','ragender','rabyear'] 

cleaned['ragender'] = cleaned['ragender'].astype('category')
features_testpart = cleaned[feature_names_testpart] 
features_testpart = features_testpart.values
features_testpart.tolist()
labels_testpart = cleaned['cesd_score']
labels_testpart = [int(labels_testpart.ix[i]) for i in labels_testpart.index] # summary: array could list(), series no this transform
features_testpart_nn=deepcopy(features_testpart)
labels_testpart_nn=deepcopy(labels_testpart)
features_testpart_nn = np.array(features_testpart_nn)

temp = int(len(feature_names_testpart)/(ew-sw+1-1))
features_testpart_nn_varyLength = features_testpart_nn[:,-1*temp*varyLength:]

min_max_scaler = preprocessing.MinMaxScaler((0,1))
features_testpart_nn_varyLength = min_max_scaler.fit_transform(features_testpart_nn_varyLength)
training_data_testpart_x, test_data_testpart_x, training_data_testpart_y ,test_data_testpart_y = train_test_split(features_testpart_nn_varyLength, 
                                                                                                                  labels_testpart_nn,test_size = 0.99,random_state=random_seed)
test_y1 = pd.Series(test_data_testpart_y)
test_y1[test_y1<=2]=0
test_y1[test_y1>2]=1

#===write the samples to the disk
temp = pd.DataFrame(training_data_x)
temp.columns = feature_names
temp.to_csv('./sample_file/training_data_x_5_10.csv',index = False)

temp = pd.DataFrame(training_data_y)
temp.to_csv('./sample_file/training_data_y_11.csv',index = False)

temp = pd.DataFrame(test_data_testpart_x)
temp.columns = feature_names_testpart
temp.to_csv('./sample_file/test_data_x_6_11.csv',index = False)

temp = pd.DataFrame(test_data_testpart_y)
temp.to_csv('./sample_file/test_data_y_12.csv',index = False)


