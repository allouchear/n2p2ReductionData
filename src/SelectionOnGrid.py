import numpy as np
import pandas as pd
import random
import os
import sys
import argparse

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def getArguments():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--infile", type=str, default="functions.h5", help="G functions from h5 file, see buildGh5.py")
	parser.add_argument("--outfile", type=str, default="numStructs.csv", help="output file : list of selected structures")
	parser.add_argument("--method", type=str, default="PCA", help="method : PCA, or TSNE")
	parser.add_argument("--scaling", type=str, default="None", help="scaling : None, MinMax, Standard or MinAbs")
	parser.add_argument("--k", type=float, default=1, help=" k = number of reduced dimension for TSNE(from 1 to 3) or PCA. For PCA, k can be a real between 0 and 1.0")
	parser.add_argument("--p", type=float, default=0.20, help=" real : % of selected structures by cluster, default=0.20. If p<0 : m=int(-p) for each direction")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")
	args = parser.parse_args()
	return args

def printStore(store):
	print(store.keys())
	for key in store.keys(): 
		dfz = store.get(key)
		z=int(dfz.iloc[0,0])
		print("Z=",z," Shape = ",dfz.shape)
		print(dfz)

def scaleX(df, args):
	df=df.drop(columns=['Z'])
	if args.scaling.upper()=="STANDARD":
		scaler = StandardScaler()
	elif args.scaling.upper()=="MINMAX":
		scaler = StandardMinMax()
	elif args.scaling.upper()=="MINABS":
		scaler = StandardMinMax()
	else:
		return df,None
	scaler.fit(df)
	df =scaler.transform(df)
	return df,scaler

def makeSelection(store,args):
	seed = args.seed
	percentage = args.p
	sample = []
	random.seed(seed)
	print("Types = ",store.keys(),flush=True)
	for key in store.keys(): 
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("-------------------------------------------------------------------------",flush=True)
		print("Selection for atomic number Z = ",z)
		print("=====================================================",flush=True)
		print("DataFrame shape = ",df.shape)
		ndf, scaler = scaleX(df, args)
		#print("Columns = ",df.columns)
		#print(df)
		#maxG=df.max()
		#print(maxG)
		#maxG = float(max(maxG[1:-1]))
		#print(maxG)
		n_components = ndf.shape[1]
		if args.k< n_components and args.method.upper()!="NONE":
			n_components=args.k
			if args.method.upper()=="PCA":
				model = PCA(n_components=n_components)
				ndf = model.fit_transform(ndf)
				n_components=model.n_components_
			elif args.method.upper()=="TSNE":
				n_components=int(n_components)
				if n_components>3 or n_components<1:
					print("Error, k must be between 1 and 3 for TSNE")
					sys.exit()
				model = TSNE(n_components=n_components)
				ndf = model.fit_transform(ndf)
		else:
			ndf = ndf.to_numpy(dtype='float32')

		nAll=ndf.shape[0]
		if percentage>0:
			m = int((nAll/100*percentage)**(1.0/n_components))
			print("histogram size for one variable=",m)
			mall = m**n_components
			print("histogram size=",mall)
		else:
			m = int(-percentage)
			print("histogram size for one variable=",m)
			mall = m**n_components
			print("histogram size=",mall)
		cols=[]
		print("n_components =",n_components)
		#print("ndf")
		#print(ndf)
		for ic in range(n_components):
			xColName="X"+str(ic)
			kColName="K"+str(ic)
			df[xColName] = ndf[:,ic]
			xmin=df[xColName].min()
			xmax=df[xColName].max()
			print("ic= {:0.12e} xmin= {:0.12e} xmax = {:0.12e}".format(ic,xmin,xmax))
			dx = (xmax-xmin)/m;
			if abs(dx)>1e-14:
				kAll = (df[xColName]-xmin)/dx;
				df[kColName] = kAll
				df[kColName] = df[kColName].astype('int')
				cols.append(kColName)
				df.loc[df[kColName] >=m, kColName] = m-1
				#print(df[df[kColName]>=m])

		#print("remove duplicated ")
		df = df.drop_duplicates(subset=cols, keep='first') 
		#print("len after = ", len( df.index.to_list()))
		#print("df after = ",  df)
		indexes = df.index.to_list()
		#print(len(indexes))
		
		indexes = list(set(indexes))
		sample.extend(indexes)

	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample

args = getArguments()
method=args.method
infile=args.infile
outfile=args.outfile
store = pd.HDFStore(args.infile,'r')

sample = makeSelection(store,args)
store.close()

print("# of selected structures = ",len(sample))
if len(sample)<50:
	print(sample)

# to check
dfn=pd.DataFrame(sample)
dfn.to_csv(outfile,index=False)


print("number of selected structures are saved in ", outfile, " file", flush=True)

# to check
#dfnt=pd.read_csv(outfile)
#print(dfnt)

