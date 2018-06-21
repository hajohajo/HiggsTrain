import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

from sklearn.metrics import roc_curve, auc
from root_pandas import read_root




data=read_root('test_sample.root')




import cPickle
with open('BDT_500.pkl','rb') as f:
       clf = cPickle.load(f)


preds=clf.predict_proba(data.drop(['isSignal','TransMass'],axis=1))[:,1]
data['Predictions']=preds #clf.predict_proba(data.drop(['isSignal','TransMass'],axis=1))[-1]

sig=data[data['isSignal']==1]
bkg=data[data['isSignal']==0]

binning_mT=np.logspace(-1.0,4.0,num=40)
indices_sig=np.digitize(sig.TransMass,binning_mT)
indices_bkg=np.digitize(bkg.TransMass,binning_mT)

signal_weights=[np.mean(sig[indices_sig==x]['Predictions']) for x in range(40)]
std_sig = np.array([np.std(sig[indices_sig==x]['Predictions']) for x in range(40)])
std_sig[np.isnan(std_sig)]= 1
std_sig[std_sig==0]= 1

background_weights=[np.mean(bkg[indices_bkg==x]['Predictions']) for x in range(40)]
std_bkg = np.array([np.std(bkg[indices_bkg==x]['Predictions']) for x in range(40)])
std_bkg[np.isnan(std_bkg)]= 1 #np.nanmax(err_DNN)
std_bkg[std_bkg==0]= 1 #np.nanmax(err_DNN)

plt.scatter(binning_mT,signal_weights,label='Signal') #,s=8)
plt.fill_between(binning_mT,signal_weights-std_sig,signal_weights+std_sig,alpha=0.4,label='$\pm 1\sigma$',color='blue')

plt.scatter(binning_mT,background_weights,label='Background')
plt.fill_between(binning_mT,background_weights-std_bkg,background_weights+std_bkg,alpha=0.4,label='$\pm 1\sigma$',color='orange')

plt.xscale('log')
plt.ylim(0.,1.0)
plt.title('Average BDT output for true tracks')
plt.ylabel('MVA output')
plt.xlabel('m_T (GeV)')
plt.legend()
plt.grid()
plt.savefig('BDTs_vs_mT.pdf')
plt.clf()

plt.hist(sig['Predictions'],bins=np.linspace(0.0,1.0,20),label='Signal',density=True,alpha=0.8)
plt.hist(bkg['Predictions'],bins=np.linspace(0.0,1.0,20),label='Background',density=True,alpha=0.8)
plt.xlabel('BDT output')
plt.ylabel('Normalized events')
plt.legend()
plt.grid()
plt.savefig('BDT_distr.pdf')
plt.clf()
"""
for step in steps:
        print Algo.toString(step)
        if step==99:
                data_pl=data
        else:
                data_pl=data[data['trk_algo'].values==step]

        true_trks=data_pl[data_pl['trk_isTrue']==0] #1
        fake_trks=data_pl[data_pl['trk_isTrue']==1] #0

        true_trks['trk_mva_DNN']=true_trks.loc[:,'trk_mva_DNN'].apply(lambda row: 2.0*row-1.0)

        hp_cut_DNN=0.0
        hp_cut_MVA=-0.75
        binning_pt=np.logspace(-1.0,3.0,num=40)
        indices_true = np.digitize(true_trks.trk_pt,binning_pt)

        iters_=[100,2000,10000]
#       iters_=[2000]
        mva_weights_=[np.mean(true_trks[indices_true==x]['trk_mva']) for x in range(40)]
        dnn_weights_=[np.mean(true_trks[indices_true==x]['trk_mva_DNN']) for x in range(40)]
        BDT_weights_=[]
        for iter in iters_:
                BDT_weights_.append([np.mean(true_trks[indices_true==x]['BDT_'+str(iter)]) for x in range(40)])

#       plt.scatter(binning_pt,mva_weights_,label='MVA baseline')
#        plt.scatter(binning_pt,dnn_weights_,label='DNN')
        for i in range(len(iters_)):
                plt.scatter(binning_pt,BDT_weights_[i],label='BDT '+str(iters_[i]),s=8)

        plt.xscale('log')
        plt.ylim(-1.0,1.0)
        plt.title('Average BDT output for true tracks')
        plt.ylabel('MVA output')
        plt.xlabel('p_T (GeV)')
        plt.legend()
        plt.grid()
        plt.savefig('BDTs_highPtTriplet_true.pdf')
        plt.clf()



"""
