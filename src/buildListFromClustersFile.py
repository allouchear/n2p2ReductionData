import os
import sys
import numpy as np
import pandas as pd
import argparse
import random

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--infile', default='clusters.csv', type=str, help=' file containing strcutures number, Z and clusters number, procuced by Clustering.py')
	parser.add_argument('--outfile', default='numStructs.csv', type=str, help='')
	parser.add_argument("--p", type=float, default=0.2, help=" real : % of selected structures, default=0.2")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")

	args = parser.parse_args()

	return args

def buildList(dfClusters,args):
	ls = list(np.unique(dfClusters['Z'].to_numpy()))
	random.seed(args.seed)
	sample =[]
	for z in ls:
		df = dfClusters[dfClusters['Z']==z]
		#labels = list(df['predicted_cluster'].to_numpy())
		labels = (df['predicted_cluster'].to_numpy())
		k = len(set(labels[labels != -1])) # number of clusters without -1
		km1 = len(labels [labels == -1]) # number of outliers
		print("Z=",z, ", number of outliers =", km1, ", number of clusters = ", k, flush=True)
		imin=0
		if km1>0:
			imin=-1
		
		df.set_index(['ID'],inplace=True)
		#print(df)
		for i in range(imin,k):
			indexes = df.index[df['predicted_cluster'] == i].tolist()
			#print(indexes)
			indexes = list(set(indexes))
			if args.p>0:
				n = max( [int( len(indexes) * (args.p/100.0) ), 1])
			else:
				n=int(-args.p)
			if n<1 :
				n=1
			if n>len(indexes):
				n=len(indexes)
			if len(indexes)>=n:
				print("Number of selected structures for this z = ", n,flush=True)
				sample.extend(random.sample(indexes, n))

	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample



args = getArguments()
infile=args.infile 
outfile=args.outfile 
dfClusters=pd.read_csv(infile)
sample=buildList(dfClusters,args)

print("# of selected structures = ",len(sample))
if len(sample)<50:
	print(sample)

# to check
dfn=pd.DataFrame(sample)
dfn.to_csv(outfile,index=False)
print("number of selected structures are saved in ", outfile, " file", flush=True)

print("See ", outfile, ' file')

