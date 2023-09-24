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

def getArguments():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--infile", type=str, default="functions.h5", help="G functions from h5 file, see buildGh5.py")
	parser.add_argument("--outfile", type=str, default="numStructs.csv", help="output file : list of selected structures")
	parser.add_argument("--method", type=str, default="KMeans", help="method : KMeans, DBSCAN or HDBSCAN")
	parser.add_argument("--k", type=int, default=5, help=" k = number of clusters (5 = default)")
	parser.add_argument("--p", type=float, default=0.20, help=" real : % of selected structures by cluster, default=0.20")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")
	parser.add_argument("--eps", type=float, default=-0.01, help=" eps for DBSCAN, if esp <0=> kneed locater of distances from NearestNeighbors*eps. For HDBSCAN min_cluster_size=ndata*eps")
	parser.add_argument("--minsample", type=int, default=2, help=" minsample: integer, see DBSCAN method")
	args = parser.parse_args()
	return args

def printStore(store):
	print(store.keys())
	for key in store.keys(): 
		dfz = store.get(key)
		z=int(dfz.iloc[0,0])
		print("Z=",z," Shape = ",dfz.shape)
		print(dfz)

def getOptimialEpsDBSCAN(df, args):
	eps=args.eps
	minsample=args.minsample
	if eps>0:
		return eps
	print("Get optimal eps for DBSCAN using NearestNeighbors+kneedle method....",flush=True)
	df=df.drop(columns=['Z'])
	neighbors = NearestNeighbors(n_neighbors=minsample)
	neighbors_fit = neighbors.fit(df)
	distances, indices = neighbors_fit.kneighbors(df)
	distances = np.sort(distances, axis=0)
	distances = distances[:,1]
	del neighbors
	del neighbors_fit
	#print(distances)
	kneedle = KneeLocator(x = range(len(distances)), y = distances, S = 1.0, curve = "concave", direction = "increasing", online=True)
	# get the estimate of knee point
	print("optimalEps=",kneedle.knee_y,flush=True)
	return kneedle.knee_y

def DBSCANClustering(store,args):
	seed = args.seed
	percentage = args.p
	eps=args.eps
	minsample=args.minsample
	sample = []
	random.seed(seed)
	print("Types = ",store.keys(),flush=True)
	for key in store.keys(): 
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("-------------------------------------------------------------------------",flush=True)
		print("Clustering of atoms with atomic number Z = ",z)
		print("=====================================================",flush=True)
		print("DataFrame shape = ",df.shape)
		#print("Columns = ",df.columns)
		#print(df)
		#maxG=df.max()
		#print(maxG)
		#maxG = float(max(maxG[1:-1]))
		#print(maxG)
		if args.method.upper()=="HDBSCAN":
			e = eps
			if eps<0:
				e = -eps
			nc=int(df.shape[0]*e)
			if nc<2 :
				nc=2
			print("min_cluster_size = ",nc)
			dbscan = HDBSCAN(min_cluster_size=nc).fit(df)
		else:
			e=getOptimialEpsDBSCAN(df,args)
			dbscan = DBSCAN(eps=e, min_samples=minsample).fit(df)
		dbscan.fit(df)
		df["predicted_cluster"] = dbscan.labels_
		k = len(set(dbscan.labels_ [dbscan.labels_ != -1])) # number of clusters without -1
		#print(dbscan.labels_)
		km1 = len(dbscan.labels_ [dbscan.labels_ == -1]) # number of outliers
		print("Z=",z, ", number of outliers =", km1, ", number of clusters = ", k, flush=True)
		#print("Clusters # ", set(dbscan.labels_))
		#print("Labels=", dbscan.labels_)
		imin=0
		if km1>0:
			imin=-1
		
		for i in range(imin,k):
			indexes = df.index[df['predicted_cluster'] == i].tolist()
			indexes = list(set(indexes))
			n = max( [int( len(indexes) * (percentage/100.0) ), 1])
			sample.extend(random.sample(indexes, n))

	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample

def KMeansClustering(store,args):
	percentage = args.p
	seed = args.seed
	k = args.k
	if seed<0:
		seed=None
	sample = []
	random.seed(seed)
	print("Types = ",store.keys())
	for key in store.keys(): 
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("Clustering of atoms with atomic number Z = ",z,flush=True)
		kmeans = KMeans(n_clusters=k,random_state=seed,n_init='auto')
		#kmeans = KMeans(n_clusters=k,init='random',random_state=seed)
		kmeans.fit(df)
		df["predicted_cluster"] = kmeans.labels_

		for i in range(k):
			indexes = df.index[df['predicted_cluster'] == i].tolist()
			indexes = list(set(indexes))
			n = max( [int( len(indexes) * (percentage/100.0) ), 1])
			sample.extend(random.sample(indexes, n))
	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample

args = getArguments()
method=args.method
infile=args.infile
outfile=args.outfile
store = pd.HDFStore(args.infile,'r')

# printStore(store)
if method.upper()=="KMEANS":
	sample = KMeansClustering(store,args)
if method.upper()=="DBSCAN" or  method.upper()=="HDBSCAN" :
	sample = DBSCANClustering(store,args)
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

