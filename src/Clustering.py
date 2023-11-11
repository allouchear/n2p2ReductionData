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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import cpu_count

def getArguments():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--infile", type=str, default="functions.h5", help="G functions from h5 file, see buildGh5.py")
	parser.add_argument("--outfile", type=str, default="numStructs.csv", help="output file : list of selected structures")
	parser.add_argument("--outclustersfile", type=str, default="clusters.csv", help="output file containing strcutures number, Z and clusters number")
	parser.add_argument("--method", type=str, default="KMeans", help="method : KMeans, DBSCAN, HDBSCAN or None")
	parser.add_argument("--k", type=int, default=5, help=" k = number of clusters (5 = default)")
	parser.add_argument("--p", type=float, default=0.20, help=" real : % of selected structures by cluster, default=0.20")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")
	parser.add_argument("--eps", type=float, default=-0.01, help=" eps for DBSCAN, if esp <0=> kneed locater of distances from NearestNeighbors*eps. For HDBSCAN min_cluster_size=ndata*eps")
	parser.add_argument("--minsample", type=int, default=2, help=" minsample: integer, see DBSCAN method")
	parser.add_argument("--reddim", type=str, default="None", help="method : None or PCA")
	parser.add_argument("--kr", type=float, default=1, help=" kr = number of reduced dimension. kr can be a real between 0 and 1.0")
	parser.add_argument("--scaling", type=str, default="None", help="scaling : None, MinMax, Standard or MaxAbs")
	parser.add_argument("--reddata", type=str, default="None", help="Reduce data method : None(default), MaxG, StdG")
	parser.add_argument("--kreddata", type=int, default=10, help="Max number of G")
	args = parser.parse_args()
	return args

def printStore(store):
	print(store.keys())
	for key in store.keys(): 
		dfz = store.get(key)
		z=int(dfz.iloc[0,0])
		print("Z=",z," Shape = ",dfz.shape)
		print(dfz)

def getMyKneedle(y):
	x = np.arange(0, len(y))
	x = x/x.max()
	data = np.vstack([x, y])
	#print(data)
	p1 = data.T[0]
	p2 = data.T[-1]
	# we need the line passing by the first and the last point
	p2mp1 = p2 - p1

	# adding an extra dimension for broadcasting
	p1 = np.expand_dims(p1, axis=1)
	p2mp1 = np.expand_dims(p2mp1, axis=1)
	diff = p1 - data
	cross_prod = np.abs(np.cross(p2mp1, diff, axis=0))  # /norm(p2-p1) is useless in this context
	knee_position = np.argmax(cross_prod)
	knee = data[:, knee_position][1]
	return knee, knee_position


def getOptimialEpsDBSCAN(df, args):
	eps=args.eps
	minsample=args.minsample
	if eps>0:
		return eps
	print("Get optimal eps for DBSCAN using NearestNeighbors+kneedle method....",flush=True)
	#df=df.drop(columns=['Z'])
	neighbors = NearestNeighbors(n_neighbors=minsample, n_jobs=cpu_count())
	neighbors_fit = neighbors.fit(df)
	#print("End fit",flush=True)
	distances, indices = neighbors_fit.kneighbors(df)
	distances = np.sort(distances, axis=0)
	#print("End sort",flush=True)
	distances = distances[:,1]
	del neighbors
	del neighbors_fit
	ldist =len(distances)
	#print(distances,flush=True)
	#print("len distances=", ldist,flush=True)
	#print("type distances=", type(distances),flush=True)
	eps, pos = getMyKneedle(distances)
	#print("MyoptimalEps=",eps,flush=True)
	#distances=distances[pos:-1]

	distances=distances[distances>=eps/2]
	#print("Len distances after reduction=",len(distances))
	kneedle = KneeLocator(x = range(len(distances)), y = distances, S = 1.0, curve = "concave", direction = "increasing", online=True)
	eps = kneedle.knee_y

	# get the estimate of knee point
	print("optimalEps=",eps,flush=True)
	return eps

def makePipeLineScalerRed(args, n_components):
	scaler = None
	if args.scaling.upper()=="STANDARD":
		print("Data scaled using ", args.scaling.upper(), " method.",flush=True)
		scaler = StandardScaler()
	elif args.scaling.upper()=="MINMAX":
		print("Data scaled using ", args.scaling.upper(), " method.",flush=True)
		scaler = MinMaxScaler()
	elif args.scaling.upper()=="MAXABS":
		print("Data scaled using ", args.scaling.upper(), " method.",flush=True)
		scaler = MaxAbsScaler()
	else:
		print("No scaling for data.",flush=True)
		scaler = None
	reddim = None
	if args.kr< n_components and args.reddim.upper()!="NONE":
		n_components=args.kr
		if abs(n_components-1)<1e-10:
			n_components=1
		if n_components>=1:
			n_components=int(n_components)
		print("n_comps = ", n_components)
		if args.reddim.upper()=="PCA":
			print("Dimensions reduced using ", args.reddim.upper(), " method.",flush=True)
			reddim = PCA(n_components=n_components)
		else:
			reddim = None

	pipe = None
	if scaler is not None and reddim is not None:
		pipe = Pipeline([('scaler', scaler), ('reddim', reddim)])
	elif scaler is not None:
		pipe = Pipeline([('scaler', scaler)])
	elif reddim is not None:
		pipe = Pipeline([('reddim', reddim)])
	return pipe

def getndf(pipe, df):
	ndf=df.drop(columns=['Z'])
	if pipe is None:
		return ndf, None
	params = pipe.get_params()
	lists, _ = zip(*params['steps'][:])
	#print(params)
	#print(pipe)
	#print(list1)
	if not ('scalar' in lists or 'reddim' in lists ):
		ndf = ndf.to_numpy(dtype='float32')
	return ndf, lists

def transformScalerRed(df, args):
	pipe = makePipeLineScalerRed(args, df.shape[1]-1) # to remove 'Z'
	ndf,lists = getndf(pipe, df)
	if lists is not None:
		ndf = pipe.fit_transform(ndf)
		if 'reddim' in lists:
			print("dimension after reduction = ", pipe.named_steps['reddim'].n_components_, flush=True)
		else:
			print("dimension = ", df.shape[1], flush=True)
	return ndf

def reduceData(df, args):
	if args.reddata.upper()=="NONE":
		return df
	if args.kreddata>=df.shape[0]:
		return df
		#print(df)
	if args.reddata.upper()=="MAXG" :
		print("Reduction of data using G with max value")
		dfs=df.max().to_frame()[1:]
	else:
		print("Reduction of data using G with max STD")
		dfs=df.std().to_frame()[1:]

	#print(dfs)
	nameColG=dfs.idxmax()[0]
	print("\tnumColGMax=", nameColG)
	df.sort_values(by=[nameColG], inplace=True, ascending=False)
	ns=df.shape[0]
	nsel=int(args.kreddata)
	nstep=ns//nsel
	listi=list(range(0,ns))
	listi=listi[0::nstep]
	if len(listi)>nsel:
		nstep +=1
		nsel=int(ns//nstep)
		listi=list(range(0,ns))
		listi=listi[0::nstep]
	print("\tTotal number of G=",ns)
	print("\tNumber of selected G=",nsel)
	print("\tnStep=",nstep)
	#print(len(listi))
	#print(listi)
	print("\tshape before reduction of data ",df.shape)
	df = df.iloc[listi]
	#print(df)
	print("\tshape after reduction of data = ",df.shape)
	return df


def DBSCANClustering(store,args):
	seed = args.seed
	percentage = args.p
	eps=args.eps
	minsample=args.minsample
	sample = []
	random.seed(seed)
	print("Types = ",store.keys(),flush=True)
	dfClusters=None
	for key in store.keys(): 
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("-------------------------------------------------------------------------",flush=True)
		print("Clustering of atoms with atomic number Z = ",z)
		print("=====================================================",flush=True)
		df = reduceData(df, args)
		print("DataFrame shape = ",df.shape, flush=True)
		print("Number of variables = ",df.shape[1]-1, flush=True)
		#print("Columns = ",df.columns)
		#print(df)
		#maxG=df.max()
		#print(maxG)
		#maxG = float(max(maxG[1:-1]))
		#print(maxG)
		ndf = transformScalerRed(df, args)
		if args.method.upper()=="HDBSCAN":
			e = eps
			if eps<0:
				e = -eps
			nc=int(df.shape[0]*e)
			if nc<2 :
				nc=2
			print("min_cluster_size = ",nc,flush=True)
			print("Begin HDBSCAN ...",flush=True)
			dbscan = HDBSCAN(min_cluster_size=nc, n_jobs=cpu_count()).fit(ndf)
			print("End HDBSCAN ",flush=True)
		else:
			e=getOptimialEpsDBSCAN(ndf,args)
			print("Begin DBSCAN ...",flush=True)
			dbscan = DBSCAN(eps=e, min_samples=minsample,n_jobs=cpu_count()).fit(ndf)
			print("End DBSCAN ",flush=True)

		#dbscan.fit(df)
		df["predicted_cluster"] = dbscan.labels_

		dfc=df[["Z","predicted_cluster"]]
		if dfClusters is None:
			dfClusters=dfc
		else:
			dfClusters=pd.concat([dfClusters,dfc])
		k = len(set(dbscan.labels_ [dbscan.labels_ != -1])) # number of clusters without -1
		#print(dbscan.labels_)
		km1 = len(dbscan.labels_ [dbscan.labels_ == -1]) # number of outliers
		print("Z=",z, ", number of outliers =", km1, ", number of clusters = ", k, flush=True)
		#print("Clusters # ", set(dbscan.labels_))
		#print("Labels=", dbscan.labels_)
		imin=0
		if km1>0:
			imin=-1
		
		nAll = 0
		for i in range(imin,k):
			indexes = df.index[df['predicted_cluster'] == i].tolist()
			indexes = list(set(indexes))
			if percentage>0:
				n = max( [int( len(indexes) * (percentage/100.0) ), 1])
			else:
				n=int(-percentage)
			if n<1 :
				n=1
			if n>len(indexes):
				n=len(indexes)
			if i<0:
				n=len(indexes)
			if len(indexes)>=n:
				print("Number of selected structures from cluster {:3d}  for this z = {:3d} from {:5d}".format(i, n,len(indexes)),flush=True)
				sample.extend(random.sample(indexes, n))
				nAll += n
		print("Total number of selected structures for this z = {:3d}".format(nAll),flush=True)

	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample, dfClusters

def KMeansClustering(store,args):
	percentage = args.p
	seed = args.seed
	k = args.k
	if seed<0:
		seed=None
	sample = []
	random.seed(seed)
	print("Types = ",store.keys())
	dfClusters=None
	for key in store.keys(): 
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("-------------------------------------------------------------------------",flush=True)
		print("Clustering of atoms with atomic number Z = ",z,flush=True)
		print("=====================================================",flush=True)
		df = reduceData(df, args)
		print("DataFrame shape = ",df.shape, flush=True)
		print("Number of variables = ",df.shape[1]-1, flush=True)
		ndf = transformScalerRed(df, args)
		kmeans = KMeans(n_clusters=k,random_state=seed,n_init='auto')
		#kmeans = KMeans(n_clusters=k,init='random',random_state=seed)
		kmeans.fit(ndf)
		df["predicted_cluster"] = kmeans.labels_
		dfc=df[["Z","predicted_cluster"]]
		if dfClusters is None:
			dfClusters=dfc
		else:
			dfClusters=pd.concat([dfClusters,dfc])

		print("Number of clusters = ", k)
		for i in range(k):
			indexes = df.index[df['predicted_cluster'] == i].tolist()
			indexes = list(set(indexes))
			if percentage>0:
				n = max( [int( len(indexes) * (percentage/100.0) ), 1])
			else:
				n=int(-percentage)
			if n<1 :
				n=1
			if n>len(indexes):
				n=len(indexes)
			if len(indexes)>=n:
				print("Number of selected structures from cluster {:3d}  for this z = {:3d} ".format(i, n),flush=True)
				sample.extend(random.sample(indexes, n))
	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample, dfClusters

def NoClustering(store,args):
	percentage = args.p
	seed = args.seed
	k = args.k
	if seed<0:
		seed=None
	sample = []
	random.seed(seed)
	print("Types = ",store.keys())
	dfClusters=None
	for i, key in enumerate(store.keys()):
		df = store.get(key)
		z=int(df.iloc[0,0])
		print("-------------------------------------------------------------------------",flush=True)
		print("Without Clusterning of data for atomic number Z = ",z,flush=True)
		print("=====================================================",flush=True)
		df = reduceData(df, args)
		print("DataFrame shape = ",df.shape, flush=True)
		print("Number of variables = ",df.shape[1]-1, flush=True)

		df["predicted_cluster"] = i
		dfc=df[["Z","predicted_cluster"]]
		if dfClusters is None:
			dfClusters=dfc
		else:
			dfClusters=pd.concat([dfClusters,dfc])

		indexes = df.index.tolist()
		indexes = list(set(indexes))
		if percentage>0:
			n = max( [int( len(indexes) * (percentage/100.0) ), 1])
		else:
			n=int(-percentage)
		if n<1 :
			n=1
		if n>len(indexes):
			n=len(indexes)
		if len(indexes)>=n:
			print("Number of selected structures from this cluster, for this z = ", n,flush=True)
			sample.extend(random.sample(indexes, n))
	#print(len(sample))
	sample = set(sample)
	sample = sorted(sample)
	return sample, dfClusters


args = getArguments()
method=args.method
infile=args.infile
outfile=args.outfile
store = pd.HDFStore(args.infile,'r')

dfClusters=None
# printStore(store)
if method.upper()=="KMEANS":
	sample, dfClusters = KMeansClustering(store,args)
elif method.upper()=="DBSCAN" or  method.upper()=="HDBSCAN" :
	sample, dfClusters = DBSCANClustering(store,args)
elif method.upper()=="NONE" :
	sample, dfClusters = NoClustering(store,args)
	
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

if dfClusters is not None:
	dfClusters.to_csv(args.outclustersfile)
	print("Clusters informations are saved in ", args.outclustersfile, " file", flush=True)
