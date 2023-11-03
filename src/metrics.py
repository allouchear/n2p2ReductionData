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

def getMetricsOneZ(df, bins):
	"""Return metrics for one Z

	Parameters
	----------
	df : dataframe containing G values for one Z
	bins : the number of bins used to build histogram

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
		x2, nx2 = getX2(df[c],bins)
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
	bins=args.bins
	dfMetrics = {}
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
			dfm = getMetricsOneZ(df, bins)
			dfm['file']=f
			z=int(df.iloc[0,0])
			sz=str(z)
			if sz in dfMetrics.keys():
				dfMetrics[str(z)] = pd.concat([dfMetrics[str(z)], dfm])
			else:
				dfMetrics[str(z)] = dfm
		store.close()
		print("-----------------------------------------------------------",flush=True)
	return dfMetrics

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
dfMetrics = getMetrics(args,files)
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

