import ROOT as R
import numpy as np
import pandas as pd
import math
import os

#Handy functions for sorting names
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

#Percentage of samples to use for training
perc=0.2
isSignal=False

#Allow TTree to be larger than default (necessary)
R.TTree.SetMaxTreeSize(1000*(2000000000))

#Variables to be read from the Ntuple

#Jets_pt=0
#Jets_phi=0
#Jets_eta=0

 
R.gROOT.ProcessLine(
"struct MyStruct {\
	vector<double>*	fJets_pt=0;\
	vector<double>*	fJets_eta=0;\
	vector<double>*	fJets_phi=0;\
	vector<int>* fJets_flavour=0;\
	vector<double>* fTaus_pt=0;\
	vector<double>* fTaus_phi=0;\
	vector<double>* fTaus_eta=0;\
	vector<double>* fTaus_lChTrkPt=0;\
	vector<double>* fElectrons_pt=0;\
        vector<double>* fEle_EffAreaIso=0;\
	vector<double>* fMuons_pt=0;\
        vector<double>* fMuon_RelIso=0;\
	Double_t fMET_Type1_x=0;\
	Double_t fMET_Type1_y=0;\
        Float_t Bjet_pt=0;\
        Float_t Bjet_eta=0;\
        Float_t Bjet_phi=0;\
	Int_t nJets=0;\
        Float_t Tau_pt=0;\
        Float_t Tau_phi=0;\
        Float_t Tau_eta=0;\
        Int_t nTaus=0;\
        Float_t MET=0;\
        Float_t TransMass=0;\
        Float_t tj1Dist=0;\
	Float_t DPhi_tau_miss=0;\
        Float_t DPhi_bjet_miss=0;\
        Float_t Upsilon=0;\
};" );

from ROOT import MyStruct
mystruct = MyStruct()


#File list of samples to be read from
if isSignal:
	samples=[
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_1000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_10000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_1500',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_2000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_2500',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_3000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_5000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_7000',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_750',
'ChargedHiggs_HplusTB_HplusToTauNu_HeavyMass_M_800',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_145',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_150',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_155',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_160',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_165',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_170',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_175',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_180',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_190',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassNoNeutral_M_200',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_145',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_150',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_155',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_165',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_170',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_175',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_180',
'ChargedHiggs_HplusTB_HplusToTauNu_IntermediateMassWithNeutral_M_200',
'ChargedHiggs_HplusTB_HplusToTauNu_M_180',
'ChargedHiggs_HplusTB_HplusToTauNu_M_200',
'ChargedHiggs_HplusTB_HplusToTauNu_M_220',
'ChargedHiggs_HplusTB_HplusToTauNu_M_250',
'ChargedHiggs_HplusTB_HplusToTauNu_M_300',
'ChargedHiggs_HplusTB_HplusToTauNu_M_400',
'ChargedHiggs_HplusTB_HplusToTauNu_M_500',
'ChargedHiggs_HplusTB_HplusToTauNu_M_500_ext1',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_100',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_120',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_140',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_150',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_155',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_160',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_80',
'ChargedHiggs_TTToHplusBWB_HplusToTauNu_M_90',
		]
else:
	samples=[
'DYJetsToLL_M_50_ext1',
'QCD_HT1000to1500',
'QCD_HT1000to1500_ext1',
'QCD_HT100to200',
'QCD_HT1500to2000',
'QCD_HT1500to2000_ext1',
'QCD_HT2000toInf',
'QCD_HT2000toInf_ext1',
'QCD_HT200to300',
'QCD_HT200to300_ext1',
'QCD_HT300to500',
'QCD_HT300to500_ext1',
'QCD_HT500to700',
'QCD_HT500to700_ext1',
'QCD_HT50to100',
'QCD_HT700to1000',
'QCD_HT700to1000_ext1',
'ST_s_channel_4f_InclusiveDecays',
'ST_t_channel_antitop_4f_inclusiveDecays',
'ST_t_channel_top_4f_inclusiveDecays',
'ST_tW_antitop_5f_inclusiveDecays',
'ST_tW_antitop_5f_inclusiveDecays_ext1',
'ST_tW_top_5f_inclusiveDecays',
'ST_tW_top_5f_inclusiveDecays_ext1',
'TT',
'WJetsToLNu',
'WJetsToLNu_ext2',
'WJetsToLNu_HT_100To200',
'WJetsToLNu_HT_100To200_ext1',
'WJetsToLNu_HT_100To200_ext2',
'WJetsToLNu_HT_1200To2500',
'WJetsToLNu_HT_1200To2500_ext1',
'WJetsToLNu_HT_200To400',
'WJetsToLNu_HT_200To400_ext1',
'WJetsToLNu_HT_200To400_ext2',
'WJetsToLNu_HT_2500ToInf',
'WJetsToLNu_HT_2500ToInf_ext1',
'WJetsToLNu_HT_400To600',
'WJetsToLNu_HT_400To600_ext1',
'WJetsToLNu_HT_600To800',
'WJetsToLNu_HT_600To800_ext1',
'WJetsToLNu_HT_70To100',
'WJetsToLNu_HT_800To1200',
'WJetsToLNu_HT_800To1200_ext1',
'WWTo2L2Nu',
'WWTo4Q',
'WWToLNuQQ',
'WZ',
'WZ_ext1',
'ZZ',
'ZZ_ext1'
		]

#samples = ['ChargedHiggs_HplusTB_HplusToTauNu_M_180']

#Path to the multicrab
multicrabpath='/work/data/multicrab_SignalAnalysis_v8030_20180508T1342/'

#Using ROOT TChain for fast reading of .root files
chain = R.TChain("Events")

forTrain=[]
forAnalysis=[]
for folder in samples:
	path=multicrabpath+folder+"/results/"
	files = [path+x for x in os.listdir(path) if 'histograms' in x]
	files.sort(key=natural_keys)
	use = int(perc*len(files))
	if use<1:
		use=0
	for i in range(0,use):
		forTrain.append(files[i])
		chain.Add(files[i])
	for i in range(use,len(files)):
		forAnalysis.append(files[i])

chain.SetBranchStatus("*",0)

chain.SetBranchStatus("Jets_pt",1)
chain.SetBranchStatus("Jets_eta",1)
chain.SetBranchStatus("Jets_phi",1)
chain.SetBranchStatus("Jets_hadronFlavour",1);
chain.SetBranchStatus("Taus_pt",1);
chain.SetBranchStatus("Taus_phi",1);
chain.SetBranchStatus("Taus_eta",1);
chain.SetBranchStatus("Taus_lChTrkPt",1);
chain.SetBranchStatus("Electrons_pt",1);
chain.SetBranchStatus("Electrons_phi",1);
chain.SetBranchStatus("Electrons_eta",1);
chain.SetBranchStatus("Electrons_effAreaIsoDeltaBeta",1);
chain.SetBranchStatus("Muons_pt",1);
chain.SetBranchStatus("Muons_phi",1);
chain.SetBranchStatus("Muons_eta",1);
chain.SetBranchStatus("Muons_relIsoDeltaBeta04",1);
chain.SetBranchStatus("MET_Type1_x",1);
chain.SetBranchStatus("MET_Type1_y",1);

chain.SetBranchAddress("Jets_pt",R.AddressOf(mystruct,'fJets_pt'))
chain.SetBranchAddress("Jets_eta",R.AddressOf(mystruct,'fJets_eta'))
chain.SetBranchAddress("Jets_phi",R.AddressOf(mystruct,'fJets_phi'))
chain.SetBranchAddress("Jets_hadronFlavour",R.AddressOf(mystruct,'fJets_flavour'))


chain.SetBranchAddress("Taus_pt",R.AddressOf(mystruct,'fTaus_pt'))
chain.SetBranchAddress("Taus_phi",R.AddressOf(mystruct,'fTaus_phi')) 
chain.SetBranchAddress("Taus_eta",R.AddressOf(mystruct,'fTaus_eta'))
chain.SetBranchAddress("Taus_lChTrkPt",R.AddressOf(mystruct,'fTaus_lChTrkPt'))
chain.SetBranchAddress("Electrons_pt",R.AddressOf(mystruct,'fElectrons_pt')) 
chain.SetBranchAddress("Electrons_effAreaIsoDeltaBeta",R.AddressOf(mystruct,'fEle_EffAreaIso'))
chain.SetBranchAddress("Muons_pt",R.AddressOf(mystruct,'fMuons_pt')) 
chain.SetBranchAddress("Muons_relIsoDeltaBeta04",R.AddressOf(mystruct,'fMuon_RelIso'))
chain.SetBranchAddress("MET_Type1_x",R.AddressOf(mystruct,'fMET_Type1_x')) 
chain.SetBranchAddress("MET_Type1_y",R.AddressOf(mystruct,'fMET_Type1_y'))

chain.LoadTree(0)

if isSignal:
	name="Signal.root"
else:
	name="Background.root"

preprocfile = R.TFile(name,"recreate")

newtree = R.TTree("Events","Preprocessed TTree")

newtree.Branch("Tau_pt",R.AddressOf(mystruct,'Tau_pt'),"Tau_pt/F");
newtree.Branch("Bjet_pt",R.AddressOf(mystruct,'Bjet_pt'),"Bjet_pt/F");
newtree.Branch("MET",R.AddressOf(mystruct,'MET'),"MET/F");
newtree.Branch("DPhi_tau_miss",R.AddressOf(mystruct,'DPhi_tau_miss'),"DPhi_tau_miss/F");
newtree.Branch("DPhi_bjet_miss",R.AddressOf(mystruct,'DPhi_bjet_miss'),"DPhi_bjet_miss/F");
newtree.Branch("Dist_tau_bjet",R.AddressOf(mystruct,'tj1Dist'),"tj1Dist/F");
newtree.Branch("Upsilon",R.AddressOf(mystruct,'Upsilon'),"Upsilon/F");
newtree.Branch("TransMass",R.AddressOf(mystruct,'TransMass'),"TransMass/F");


entries = int(chain.GetEntries())
chain.GetEntry(0)


for i in range(1,entries):
	if(i%10000==0):
		print i,"/",entries
	chain.GetEntry(i)

        MET=np.sqrt(np.power((mystruct.fMET_Type1_x),2)+np.power((mystruct.fMET_Type1_y),2));

        #skip conditions
        if((mystruct.fJets_pt).size()<4):
        	continue;
	#Bjet
	bInd=-1
	for i in range((mystruct.fJets_pt).size()):
		if (((mystruct.fJets_flavour)[i]==5)& (bInd==-1)):
			bInd=i
	if bInd==-1:
		continue;

        if(MET<80):
        	continue;

        #Lepton isolations
	for i in range(len((mystruct.fEle_EffAreaIso))):
                if((mystruct.fEle_EffAreaIso)[i]>0.15):
        		continue;

	for i in range(len((mystruct.fMuon_RelIso))):
        	if((mystruct.fMuon_RelIso)[i]>0.15):
                	continue;


        METphi=math.atan2(mystruct.fMET_Type1_y,mystruct.fMET_Type1_x)

	mystruct.Tau_pt=(mystruct.fTaus_pt)[0]

        mystruct.Bjet_pt=(mystruct.fJets_pt)[bInd]

	mystruct.MET=MET

	mystruct.DPhi_tau_miss = min(np.abs((mystruct.fTaus_phi)[0]-METphi),2*math.pi-np.abs((mystruct.fTaus_phi)[0]-METphi))
	mystruct.DPhi_bjet_miss = min(np.abs((mystruct.fJets_phi)[bInd]-METphi),2*math.pi-np.abs((mystruct.fJets_phi)[bInd]-METphi))

	mystruct.tj1Dist=math.sqrt(np.power(min(np.abs((mystruct.fJets_phi)[bInd]-mystruct.fTaus_phi[0]),2*math.pi-np.abs((mystruct.fJets_phi)[bInd]-mystruct.fTaus_phi[0]))+np.abs(mystruct.fJets_eta[bInd]-mystruct.fTaus_eta[0]),2));

	mystruct.Upsilon=2.0*mystruct.fTaus_lChTrkPt[0]/mystruct.fTaus_pt[0]-1.0

	mystruct.TransMass=math.sqrt(2.*mystruct.Tau_pt*MET*(1.-math.cos(METphi+mystruct.Tau_phi)));


	newtree.Fill()

newtree.Write("",R.TObject.kWriteDelete)
preprocfile.Close()

#Save the filenames used for training and that can be used for analysis
file = open('trainFiles.txt','w')
for item in forTrain:
	file.write("%s\n" % item)
file.close()

file = open('analysisFiles.txt','w')
for item in forAnalysis:
	file.write("%s\n" % item)
file.close()
