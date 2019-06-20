#! /usr/bin/env python
#coding=utf-8
#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score 
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")
#%%
#=======SETTING======= 
no_auxi_input = False
Bidirectional = False
random_seed = 1
sw=5
ew=11
varyLength = 6

#=====READ DATA=======
feature_names_trainx, feature_names_testx = [], []
for i in range(6,0,-1):
    feature_names_trainx = feature_names_trainx +['r{0}shlt'.format(ew-i),'r{0}mobila'.format(ew-i),'r{0}lgmusa'.format(ew-i),'r{0}smoken'.format(ew-i),
                                                  'r{0}diab'.format(ew-i),'r{0}cancre'.format(ew-i),'r{0}cesd'.format(ew-i),'r{0}sayret'.format(ew-i),'nsp','ragender','rabyear'] 
for i in range(6,0,-1):
    feature_names_testx = feature_names_testx +['r{0}shlt'.format(ew+1-i),'r{0}mobila'.format(ew+1-i),'r{0}lgmusa'.format(ew+1-i),'r{0}smoken'.format(ew+1-i),
                                                  'r{0}diab'.format(ew+1-i),'r{0}cancre'.format(ew+1-i),'r{0}cesd'.format(ew+1-i),'r{0}sayret'.format(ew+1-i),'nsp','ragender','rabyear']
   
shlt_trainx     = [i for i in feature_names_trainx if 'shlt' in i]
mobila_trainx   = [i for i in feature_names_trainx if 'mobila' in i]
lgmusa_trainx   = [i for i in feature_names_trainx if 'lgmusa' in i]
smoken_trainx   = [i for i in feature_names_trainx if 'smoken' in i]
diab_trainx     = [i for i in feature_names_trainx if 'diab' in i]
cancre_trainx   = [i for i in feature_names_trainx if 'cancre' in i]
cesd_trainx     = [i for i in feature_names_trainx if 'cesd' in i]
sayret_trainx   = [i for i in feature_names_trainx if 'sayret' in i]
nsp_trainx      = ['nsp','nsp.1','nsp.2','nsp.3','nsp.4','nsp.5']
nsp_trainx = nsp_trainx[6-varyLength:]
ragender_trainx = ['ragender','ragender.1','ragender.2','ragender.3','ragender.4','ragender.5']
ragender_trainx = ragender_trainx[6-varyLength:]
age_trainx =      ['rabyear','rabyear.1','rabyear.2','rabyear.3','rabyear.4','rabyear.5']
age_trainx = age_trainx[6-varyLength:]

shlt_testx     = [i for i in feature_names_testx if 'shlt' in i]
mobila_testx   = [i for i in feature_names_testx if 'mobila' in i]
lgmusa_testx   = [i for i in feature_names_testx if 'lgmusa' in i]
smoken_testx   = [i for i in feature_names_testx if 'smoken' in i]
diab_testx     = [i for i in feature_names_testx if 'diab' in i]
cancre_testx   = [i for i in feature_names_testx if 'cancre' in i]
cesd_testx     = [i for i in feature_names_testx if 'cesd' in i]
sayret_testx   = [i for i in feature_names_testx if 'sayret' in i]
nsp_testx      = ['nsp','nsp.1','nsp.2','nsp.3','nsp.4','nsp.5']
nsp_trainx = nsp_trainx[6-varyLength:]

ragender_testx = ['ragender','ragender.1','ragender.2','ragender.3','ragender.4','ragender.5']
ragender_trainx = ragender_trainx[6-varyLength:]

age_testx =      ['rabyear','rabyear.1','rabyear.2','rabyear.3','rabyear.4','rabyear.5']
age_trainx = age_trainx[6-varyLength:]

droplist_trainx = [shlt_trainx, mobila_trainx, lgmusa_trainx, smoken_trainx, diab_trainx, cancre_trainx, cesd_trainx, sayret_trainx, nsp_trainx, ragender_trainx,age_trainx]
droplist_testx =  [shlt_testx, mobila_testx, lgmusa_testx, smoken_testx, diab_testx, cancre_testx, cesd_testx, sayret_testx, nsp_testx, ragender_testx,age_testx]

column_number = [2] ###use m=-1 and m=-2 to represent the last two variables. i.e. auxi_input

training_data_x = pd.read_csv('./sample_file/training_data_x_5_10.csv')
for m in column_number: training_data_x = training_data_x.drop(droplist_trainx[m], axis = 1)
variables_name = list(training_data_x.columns)
training_data_x = training_data_x.values
temp = len(droplist_trainx)-len(column_number) #minus 1 since lgmusa is not used anymore
training_data_x = training_data_x[:,-1*temp*varyLength:]



training_data_y = pd.read_csv('./sample_file/training_data_y_11.csv')

test_data_testpart_x = pd.read_csv('./sample_file/test_data_x_6_11.csv')
for m in column_number: test_data_testpart_x = test_data_testpart_x.drop(droplist_testx[m], axis = 1)
feature_names_testpart = list(test_data_testpart_x.columns)
test_data_testpart_x = test_data_testpart_x.values
temp = len(droplist_trainx)-len(column_number) #minus 1 since lgmusa is not used anymore
test_data_testpart_x = test_data_testpart_x[:,-1*temp*varyLength:]

test_data_testpart_y = pd.read_csv('./sample_file/test_data_y_12.csv')

#==
train_y1 = deepcopy(training_data_y)
train_y1[train_y1<=2]=0
train_y1[train_y1>2]=1
train_y1 = train_y1.values
train_y1 = train_y1[:,0]
training_data_y = training_data_y.values
training_data_y = list(training_data_y[:,0])
test_y1 = deepcopy(test_data_testpart_y)
test_y1[test_y1<=2]=0
test_y1[test_y1>2]=1
test_y1 = test_y1.values
test_y1 = test_y1[:,0]
test_data_testpart_y = test_data_testpart_y.values
test_data_testpart_y = list(test_data_testpart_y[:,0])

from sklearn.metrics import mean_absolute_error
from keras.layers import Dense,Input,concatenate
from keras.layers import LSTM, Dropout, Bidirectional
from keras.models import Model
from numpy.random import seed
seed(2)
#=========auxiliary lstm start =============#
n_hours = varyLength
n_features = int(len(feature_names_testpart)/(ew-sw+1-1)) #feature_names_testpart already drop

###lstm.reshape input to be 3D [samples, timesteps, features]
train_X = training_data_x.reshape((training_data_x.shape[0], n_hours, n_features))
test_X = test_data_testpart_x.reshape((test_data_testpart_x.shape[0], n_hours, n_features))

#==remove age and gender as they are auxilary inputs
train_X = train_X[:,:,:n_features-2]
test_X= test_X[:,:,:n_features-2]

n_features = n_features-2
#==========architecture=========
main_input = Input(shape=(n_hours,n_features), name='m_input')
if Bidirectional:  lstm_out = Bidirectional(LSTM(8))(main_input)
else:              lstm_out = LSTM(8)(main_input)
dout = Dropout(0.1)(lstm_out)
if (-1 in column_number) or (-2 in column_number): auxiliary_input = Input(shape=(1,), name='aux_input')
else:               auxiliary_input = Input(shape=(2,), name='aux_input')

if no_auxi_input: x = dout
else:             x = concatenate([dout, auxiliary_input])      
x = Dense(16)(x)
o1 = Dense(16,activation='relu')(x)
o2 = Dense(16,activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='m_output')(o1)
auxiliary_output = Dense(1, name='aux_output')(o2) 

#==========compile and fit=========
if no_auxi_input:
    model = Model(inputs=main_input, 
                  outputs=[main_output, auxiliary_output])
    model.compile(optimizer='adam', loss=['binary_crossentropy','mse'],
                  loss_weights=[1.0, 1.0]) 
    train_y1 = np.array(train_y1)
    train_y2 = np.array(training_data_y)
    test_y1 = np.array(test_y1)
    test_y2 = np.array(test_data_testpart_y)
    history = model.fit(train_X, [train_y1, train_y2], validation_data=(test_X, [test_y1, test_y2]),
                        epochs=30, batch_size=128)    
    yhat = model.predict(test_X) 
else:
    model = Model(inputs=[main_input, auxiliary_input], 
                  outputs=[main_output, auxiliary_output])
    model.compile(optimizer='adam', loss=['binary_crossentropy','mse'],
                  loss_weights=[1.0, 1.0])
    if (-1 in column_number) or (-2 in column_number): temp, temptest = training_data_x[:,-1:], test_data_testpart_x[:,-1:]
    else:                temp, temptest = training_data_x[:,-2:], test_data_testpart_x[:,-2:]
    temp, temptest = np.array(temp), np.array(temptest)
    if no_auxi_input: temp, temptest = np.zeros(temp.shape),np.zeros(temptest.shape)
    train_y1 = np.array(train_y1)
    train_y2 = np.array(training_data_y)
    test_y1 = np.array(test_y1)
    test_y2 = np.array(test_data_testpart_y)
    history = model.fit([train_X, temp], [train_y1, train_y2], validation_data=([test_X, temptest], [test_y1, test_y2]),
                        epochs=30, batch_size=128)  
    yhat = model.predict([test_X, temptest]) 

#=============evaluate===========
#evaluate regression
yhat = yhat[1]
yhat_ary=[]
for i in yhat:
    for j in i:
        yhat_ary.append(j)
yhat_ary = np.array(yhat_ary)
yhat_ary = yhat_ary.reshape(-1,1)
print('Test MAE: %.3f' % mean_absolute_error(test_y2, yhat_ary))

#evaluate classification
ans=[] 
if no_auxi_input: yhat = model.predict(test_X) 
else:             yhat = model.predict([test_X, temptest]) 
column=0
for i in yhat[column]:
    ans.append(i[0])  
ans = np.array(ans)
report = ans>0.5
print ('c-statistic:' , round(roc_auc_score(test_y1, ans.tolist()),3))
#==========end================

#==========lasso for regression============#
lassocv=Lasso(alpha = 0.05)
lassocv.fit(training_data_x, training_data_y)
answer = lassocv.predict(test_data_testpart_x)     
answer = answer.reshape(-1,1)
answer_ary=[]
for i in answer:
    for j in i:
        answer_ary.append(j)
print('lasso MAE: %.3f' % mean_absolute_error(test_data_testpart_y, answer_ary))
#coeficients
variables_coef1 = dict()
for i, j in zip(variables_name, lassocv.coef_ ):
    variables_coef1[i] = j


#======MLP classifier=====#
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,20), random_state=1,max_iter=50)
clf.fit(training_data_x, train_y1)
answer = clf.predict_proba(test_data_testpart_x)[:,1] 
report = answer >0.5
print ( 'mlp_classifier:', round(roc_auc_score(test_y1, answer),3) )

#========SVM classifier========#
clf = SVC( tol=0.001, probability=True,kernel = 'linear')
clf.fit(training_data_x, train_y1)  
y_pred = clf.predict_proba(test_data_testpart_x)[:,1] 
report = y_pred >= 0.5
print ('SVM:', round(roc_auc_score(test_y1, y_pred),3) )
svc_coef =  clf.coef_[0,:]
#coeficients
clf = LinearSVC( tol=0.001)   
clf.fit(training_data_x, train_y1)  
variables_coef2 = dict()
for i, j in zip(variables_name, svc_coef):
    variables_coef2[i] = j
    

