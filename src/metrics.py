import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import chisquare
import glob

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--outfile", type=str, default="metrics", help="prefix of outfiles : one .h file for all Z and one .csv for each Z")
	parser.add_argument("--bins", type=int, default=100, help=" bins = number of bins to build the histograme (100 = default)")
	parser.add_argument("--infiles", type=str, default="funct*h5", help="--infiles=list of file. Example --infiles=func*h5,a.h5")
	parser.add_argument("--p", type=float, default=-10, help=" real : number of values/number of all bins(N Dimension)*100, default=-10. If p<0 : m=int(-p) bins for each direction")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")

	args = parser.parse_args()

	return args

def getX2(dfc,bins):
	arr=dfc.to_numpy()
	h=np.histogram(arr,bins=bins)
	h = h[0]
	x2=chisquare(h)
	x2=x2.statistic
	nx2=x2/np.sum(h)
	return x2, nx2

def getChi2nDim(df, args):
	"""Return Chi2 uzing all Z

	Parameters
	----------
	df : dataframe containing G values for one Z
	args
	"""
	ndf=df
	n_components = df.shape[1]-1
	nAll=ndf.shape[0]
	if args.p>0:
		m = int((nAll/100*args.p)**(1.0/n_components))
		if m<2:
			m = 2
		print("histogram size for one variable=",m)
		mall = m**n_components
		print("histogram size=",mall)
	else:
		m = int(-args.p)
		print("histogram size for one variable=",m)
		mall = m**n_components
		print("histogram size=",mall)
	cols=[]
	print("n_components =",n_components)
	for ic in range(n_components):
		xColName="G"+str(ic+1)
		kColName="K"+str(ic+1)
		xmin=df[xColName].min()
		xmax=df[xColName].max()
		dx = (xmax-xmin)/m;
		#print("ic= {:0.12e} xmin= {:0.12e} xmax = {:0.12e} dx = {:0.12e}".format(ic+1,xmin,xmax,dx))
		if abs(dx)>1e-14:
			kAll = (df[xColName]-xmin)/dx;
			df[kColName] = kAll
			df[kColName] = df[kColName].astype('int')
			cols.append(kColName)
			df.loc[df[kColName] >=m, kColName] = m-1
			#print(df[df[kColName]>=m])

	df = df.groupby(by=cols).aggregate(['count'])['G1']
	h = df.reset_index()["count"].values
	nAll=np.sum(h)
	aver=nAll/m**n_components
	#s = aver*m**n_cmpounts # all cas are 0 
	s = nAll # all cas are 0 
	lh = len(h)
	s -= lh*aver # for case with n>0, remove aver
	haver= [aver]*lh
	s += ((h-haver)**2/haver).sum()
	chi2 = s/nAll
	print("chi2=",chi2,'m=',m)
	# now compute chi2 using only occuped sites
	aver=nAll/lh
	haver= [aver]*lh
	rchi2 = ((h-haver)**2/haver).sum()/nAll
	print("chi2 computed using occuped sites =",rchi2)
		
	return chi2, rchi2



def getMetricsOneZ(df, args):
	"""Return metrics for one Z

	Parameters
	----------
	df : dataframe containing G values for one Z
	args : 

	Returns
	-------
	dfMetrics a data frame containing 
	min: min value for each column of G
	max: max value for each column of G
	std: std value for each column of G
	X2 : normalized X2 for each column of G. The histograme is defided by the nombre of values of G
	"""
	dfm = df.aggregate(['count','min', 'max', 'mean', 'std'])
	x2l = []
	nx2l = []
	R = []
	for c in df.columns:
		#print(c)
		x2, nx2 = getX2(df[c],args.bins)
		x2l += [x2] 
		nx2l += [nx2] 
		R += [ df[c].max()-df[c].min() ]
	df2 = pd.DataFrame([R,x2l,nx2l], columns=df.columns, index=['R','x2','nx2'])
	dfMetrics = pd.concat([dfm,df2])
	listCG = [i for i in dfMetrics.columns if "G" in i]
	#print("listCG=",listCG)
	dfMetrics["Mean"] = dfMetrics[listCG].mean(axis=1)
	dfMetrics["Min"] = dfMetrics[listCG].min(axis=1)
	dfMetrics["Max"] = dfMetrics[listCG].max(axis=1)
	
	dfMetrics.drop(columns=['Z'], inplace=True)

	return dfMetrics

def getMetrics(args,files):
	dfMetrics = {}
	dfChi2nDim = {}
	for f in files:
		store = pd.HDFStore(f,'r')
		st = 'file name = {:s}'.format(f)
		print(st)
		for i in range(len(st)):
			print("=", end='')
		print(flush=True)
		for key in store.keys(): 
			print("Type = ",key)
			df = store.get(key)
			dfm = getMetricsOneZ(df, args)
			dfm['file']=f
			z=int(df.iloc[0,0])
			sz=str(z)
			if sz in dfMetrics.keys():
				dfMetrics[str(z)] = pd.concat([dfMetrics[str(z)], dfm])
			else:
				dfMetrics[str(z)] = dfm
			dfc = pd.DataFrame({"file":[f]})
			chi2, rchi2 = getChi2nDim(df, args)
			dfc['Chi2nDim']=chi2
			dfc['RChi2nDim']=rchi2
			if sz in dfChi2nDim.keys():
				dfChi2nDim[str(z)] = pd.concat([dfChi2nDim[str(z)],dfc])
			else:
				dfChi2nDim[str(z)] = dfc
		store.close()
		print("-----------------------------------------------------------",flush=True)
	return dfMetrics,  dfChi2nDim

def saveByMean(dfMetrics,args, key='nx2'):
	soutfile=os.path.splitext(args.outfile)[0]+'_sorted'+'_'+key+'.txt'
	f=open(soutfile, 'w')
	for sz in dfMetrics.keys():
		print("Z=",sz,file=f)
		print("=======",file=f)
		print(dfMetrics[sz].loc[key].sort_values(by=["Mean"], ascending=True).to_string(),file=f)
		for i in range(140):
			print("-", end='',file=f)
		print(file=f)
	f.close()
	print("Data sorted by ", key, " are saved in ", soutfile, " file", flush=True)
	print("================================================================", flush=True)

def saveByChi2nDim(dfChi2nDim,args):
	key="Chi2nDim"
	key2="RChi2nDim"
	soutfile=os.path.splitext(args.outfile)[0]+'_sorted'+'_'+key+'.txt'
	f=open(soutfile, 'w')
	print(" =======================================================================",file=f)
	print(" Chi2nDim   : Chi2 computed using all cells including those with 0 value",file=f)
	print(" RChi2nDim  : Chi2 computed using lonly cells with 1 or more values",file=f)
	print(" =======================================================================",file=f)
	for sz in  dfChi2nDim.keys():
		print("Z=",sz,file=f)
		print("=======",file=f)
		#print( dfChi2nDim[sz].reset_index()[[key,key2,'file']].drop_duplicates(subset='file', keep='first').sort_values(by=[key], ascending=True).to_string(),file=f)
		print( dfChi2nDim[sz].reset_index()[[key,key2,'file']].drop_duplicates(subset='file', keep='first').sort_values(by=[key2], ascending=True).to_string(),file=f)
		for i in range(140):
			print("-", end='',file=f)
		print(file=f)
	f.close()
	print("Data sorted by ", key, " are saved in ", soutfile, " file", flush=True)
	print("=========================================================================", flush=True)

args = getArguments()
infiles=args.infiles
infiles=infiles.split(',')
files=[]
for f in infiles:
	files += glob.glob(f)
print(files)
if len(files)<1:
	print("Error : check --infiles parameters ")
	sys.exit(1)
dfMetrics, dfChi2nDim = getMetrics(args,files)
outfile=args.outfile 
h5outfile=os.path.splitext(args.outfile)[0]+'.h5'
storeOut = pd.HDFStore(h5outfile,"w")
for sz in dfMetrics.keys():
	csvoutfile=os.path.splitext(args.outfile)[0]+'_Z'+sz+'.csv'
	print("Z=",sz)
	print("=======")
	st = 'Data for Z = {:s} saved in {:s} file'.format(sz,csvoutfile)
	print(st, flush=True)
	for i in range(len(st)):
		print("-", end='')
	print()
	print(dfMetrics[sz].to_string())
	storeOut.put('/Z_'+sz, dfMetrics[sz])
	dfMetrics[sz].to_csv(csvoutfile)
	for i in range(140):
		print("-", end='')
	print()
storeOut.close()

print("All data are in saved in ", h5outfile, " file", flush=True)
print("============================================", flush=True)

saveByMean(dfMetrics,args, key='nx2')
saveByMean(dfMetrics,args, key='R')
saveByMean(dfMetrics,args, key='std')

saveByChi2nDim(dfChi2nDim,args)
