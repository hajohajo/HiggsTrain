import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pylab as P
import pandas as pd
from root_pandas import read_root,to_root
import ROOT
import numpy as np
import glob
import math
from tqdm import tqdm
import sys,re
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

np.random.seed(7)

sig='Signal_07062018.root'
bkg='Background_07062018.root'

sig=read_root(sig)
bkg=read_root(bkg)
sig=sig.sample(n=bkg.shape[0])
sig['isSignal']=1
bkg['isSignal']=0

data=pd.concat([sig,bkg])
data=data.sample(frac=1.)
data.reset_index(drop=True,inplace=True)

data,test=train_test_split(data,shuffle=False,test_size=0.2)
test.to_root('test_sample.root')

#scaler1 = MinMaxScaler().fit(data.drop(['trk_isTrue','trk_mva'],axis=1))
#joblib.dump(scaler1, "scaler.pkl")

train_x = data.drop(['isSignal','TransMass'],axis=1) 
#train_x=pd.DataFrame(data=scaler1.transform(train_x),columns=train_x.columns.values)
train_y = data['isSignal']

#clf = RandomForestClassifier(n_estimators=100,verbose=1,n_jobs=-1,class_weight='balanced')
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(verbose=1,n_estimators=500)

clf.fit(train_x,train_y)

preds=clf.predict(train_x[:5])

print preds

import cPickle
with open('BDT_500.pkl','wb') as f:
        cPickle.dump(clf,f)

from skTMVA import convert_bdt_sklearn_tmva
convert_bdt_sklearn_tmva(clf,[('Tau_pt', 'F'), ('Bjet_pt', 'F'), ('MET', 'F'), ('DPhi_tau_miss', 'F'),('DPhi_bjet_miss', 'F'), ('Dist_tau_bjet', 'F'), ('Upsilon', 'F'), ('Transmass', 'F')], 'BDT_500.xml')


