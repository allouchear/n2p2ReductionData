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
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from joblib import cpu_count
from os import environ



def getArguments():
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--databasefile", type=str, default="functions.h5", help="G functions from h5 file, for the database. See buildGh5.py")
	parser.add_argument("--descfile", type=str, default="None", help="G functions from h5 file, for the new structures. See buildGh5.py")
	parser.add_argument("--maxfile", type=str, default="None", help="Read the max of KDE for each Z, from file maxfile. If None(default) the max are calculated(can be very long)")
	parser.add_argument("--outfile", type=str, default="resultsKDE.csv", help="output file : KDE for all structures in descfile")
	parser.add_argument("--reddim", type=str, default="None", help="method : None or PCA")
	parser.add_argument("--k", type=float, default=1, help=" k = number of reduced dimension PCA. k can be a real between 0 and 1.0")
	parser.add_argument("--scaling", type=str, default="None", help="scaling : None, MinMax, Standard or MaxAbs")
	parser.add_argument("--seed", type=int, default=111, help=" seed, default 111. If <0 => random_state=None")
	parser.add_argument("--bw", type=str, default='scott', help="band-width for KDE. scott,silverman, if not, use at it ")
	parser.add_argument("--rtol", type=float, default=0, help="rtol is the desired relative tolerance of the result.Default : 0")
	parser.add_argument("--leaf_size", type=int, default=40, help="leaf_size = Specify the leaf size of the underlying tree. Default : 40")
	args = parser.parse_args()
	return args

def printStore(store):
	print(store.keys())
	for key in store.keys(): 
		dfz = store.get(key)
		z=int(dfz.iloc[0,0])
		print("Z=",z," Shape = ",dfz.shape)
		print(dfz)

def makePipeLine(args, n_components):
	scaler = None
	if args.scaling.upper()=="STANDARD":
		scaler = StandardScaler()
	elif args.scaling.upper()=="MINMAX":
		scaler = MinMaxScaler()
	elif args.scaling.upper()=="MAXABS":
		scaler = MaxAbsScaler()
	else:
		scaler = None
	reddim = None
	if args.k< n_components and args.reddim.upper()!="NONE":
		n_components=args.k
		if abs(n_components-1)<1e-10:
			n_components=1
		if args.reddim.upper()=="PCA":
			reddim = PCA(n_components=n_components)
		else:
			reddim = None

	kde = None
	if args.bw.upper() =='SCOTT':
		kde = KernelDensity(kernel='gaussian', bandwidth='scott', rtol=args.rtol, leaf_size=args.leaf_size)
	elif args.bw.upper() =='SILVERMAN':
		kde = KernelDensity(kernel='gaussian', bandwidth='silverman', rtol=args.rtol, leaf_size=args.leaf_size)
	else:
		kde = KernelDensity(kernel='gaussian', bandwidth=abs(args.bw), rtol=args.rtol, leaf_size=args.leaf_size)

	pipe = None
	if scaler is not None and reddim is not None:
		pipe = Pipeline([('scaler', scaler), ('reddim', reddim), ('kde', kde)])
	elif scaler is not None:
		pipe = Pipeline([('scaler', scaler), ('kde', kde)])
	elif reddim is not None:
		pipe = Pipeline([('reddim', reddim), ('kde', kde)])
	else:
		pipe = Pipeline([('kde', kde)])
	return pipe

def getndf(pipe, df):
	ndf=df.drop(columns=['Z'])
	params = pipe.get_params()
	lists, _ = zip(*params['steps'][:])
	#print(params)
	#print(pipe)
	#print(list1)
	if not ('scalar' in lists or 'reddim' in lists ):
		ndf = ndf.to_numpy(dtype='float32')
	return ndf, lists

def fitKDE(df, args):
	pipe = makePipeLine(args, df.shape[1])
	ndf,lists = getndf(pipe, df)
	pipe.fit(ndf)
	if 'reddim' in lists:
		print("dimension after reduction = ", pipe.named_steps['reddim'].n_components_, flush=True)
	else:
		print("dimension = ", df.shape[1], flush=True)
	return pipe

def scoresKDE(pipe, df):
	#print("get scores....",flush=True)
	ndf,lists = getndf(pipe, df)
	#print("End getndf",flush=True)
	log_dens = pipe.score_samples(ndf)
	#print("End score_samples",flush=True)
	dens = np.exp(log_dens)
	#print("End dens calculation",flush=True)
	'''
	if 'reddim' in lists:
		ndf = pipe.named_steps['reddim'].transform(ndf)
		print("min df after transfomration = ",ndf.min(), flush=True)
		print("max df after transfomration = ",ndf.max(), flush=True)
	elif 'scaler' in lists:
		ndf = pipe.named_steps['scaler'].transform(ndf)
		print("min df after transfomration = ",ndf.min(), flush=True)
		print("max df after transfomration = ",ndf.max(), flush=True)
	else:
		print("min df  = ",ndf.min(), flush=True)
		print("max df  = ",ndf.max(), flush=True)
	'''
	return dens

def getNBNE(njobs, dfshape):
	m =  dfshape//njobs
	M =[m]*njobs
	M[njobs-1] += dfshape%njobs
	#print("M=",M)
	NB =[0]*njobs
	NE =[0]*njobs
	NE[0] = M[0]
	for i in range(1,njobs):
		NB[i] = NE[i-1]
		NE[i] = NB[i]+M[i]
	'''
	print("NB=",NB,flush=True)
	print("NE=",NE,flush=True)
	'''
	return NB, NE

def getScores(pipe, df):
	#print("Get scores....",flush=True)
	'''
	njobs=1
	if 'OMP_NUM_THREADS' in dict(environ).keys():
		njobs=int(environ['OMP_NUM_THREADS'])
	'''
	njobs = cpu_count()
	if df.shape[0]<= njobs:
		njobs=df.shape[0]
	print("Number of jobs = ",njobs, flush=True)
	print("Data size      = ",df.shape[0], flush=True)

	if njobs==1:
		dens =scoresKDE(pipe, df)
	else:
		NB,NE = getNBNE(njobs, df.shape[0])
		r = Parallel(n_jobs=njobs, verbose=20)(
			delayed(scoresKDE)(pipe, df[NB[i]:NE[i]]) for i in range(njobs)
		) 
		dens = np.concatenate(r)
		dens = dens.reshape(-1)
	return dens

def getMaxScores(pipe, df):
	print("Get scores for Database....",flush=True)
	print("---------------------------------------", flush=True)
	dens = getScores(pipe,df)
	#print("densshape=",dens.shape)
	#print("df=",df.shape)
	#np.set_printoptions(threshold=np.inf)
	#print("dens = ",dens,flush=True)
	maxdens=max(dens)
	mindens=min(dens)
	print("maxdens = ",maxdens,flush=True)
	print("mindens = ",mindens,flush=True)
	print("sumdens = ",sum(dens),flush=True)
	return maxdens, dens

def readMaxScores(maxfile):
	if maxfile.upper()=="NONE":
		return None
	dfmax=pd.read_csv(maxfile)
	print(dfmax)
	maxDensities=dfmax.to_dict('records')[0]
	print(maxDensities)
	return maxDensities 

def KDEAllZ(storeDatabase, storeDesc, args, maxDensities=None):
	seed = args.seed
	if seed<0:
		seed=None
	random.seed(seed)
	print("Types = ",storeDatabase.keys())
	dfAll = None
	print("=======================================", flush=True)
	listdens={}
	store =  storeDesc
	if storeDesc is None:
		store = storeDatabase
	for key in store.keys(): 
		if key in storeDatabase.keys(): 
			df = storeDatabase.get(key)
			df = df.loc[:, (df >= 1e-10).any(axis=0)]
			z=int(df.iloc[0,0])
			print("Build KDE.... for atomic number Z = ",z,flush=True)
			print("---------------------------------------", flush=True)
			pipe = fitKDE(df,args)
			#print(pipe.get_params())
			if maxDensities is None:
				maxdens, dens = getMaxScores(pipe, df)
				listdens[str(z)] = [maxdens]
			else:
				if str(z) in maxDensities.keys():
					maxdens=maxDensities[str(z)]
				else:
					print("=======================================================================",flush=True)
					print("Error. No value for Z = ",z," in file ", args.maxfile, flush=True)
					print("Values available in this file :  ",flush=True)
					print(maxDensities,flush=True)
					print("Set maxfile argument to None. The max of the densities will be computed",flush=True)
					print("=======================================================================",flush=True)
					sys.exit(1)

			if storeDesc is not None:
				df = storeDesc.get(key)
				df = df.loc[:, (df >= 1e-10).any(axis=0)]
				print("Get scores for descriptors....",flush=True)
				print("---------------------------------------",flush=True)
				dens = getScores(pipe,df)
				#print("densshape=",dens.shape)
				#print("df=",df.shape,flush=True)

			df['Dens']  = dens/maxdens
			df =  df [['Z','Dens']] 
			#df = df.groupby(level=0).sum()
			print("Scores ",flush=True)
			print("------------------",flush=True)
			print(df)
			print("=======================================", flush=True)

			if dfAll is None:
				dfAll = df
			else:
				dfAll = pd.concat([dfAll, df])
		else:
			df = storeDesc.get(key)
			z=int(df.iloc[0,0])
			print("============================================================",flush=True)
			print("Error. No data available in database for atomic number Z = ",z,flush=True)
			print("============================================================",flush=True)
			sys.exit(1)
		#print(dfAll)

	#dfAll.reset_index(inplace=True)
	#dfAll = dfAll.groupby(['ID', 'Z']).mean()
	#dfAll = dfAll.groupby(['ID']).mean()
	dfAll = dfAll.groupby(level=0).mean()
	dfAll = dfAll[["Dens","Z"]]
	dfAll.reset_index(inplace=True)
	dfAll = dfAll[["ID","Dens"]]
	print("Final density by system",flush=True)
	print("-----------------------",flush=True)
	print(dfAll,flush=True)
	print("maxdens = ",max(dfAll["Dens"]),flush=True)
	print("mindens = ",min(dfAll["Dens"]),flush=True)
	
	return dfAll, listdens
		
args = getArguments()
storeDatabase = pd.HDFStore(args.databasefile,'r')
storeDesc = None
fn=args.databasefile
if args.descfile.upper()!="NONE":
	storeDesc = pd.HDFStore(args.descfile,'r')
	fn=args.descfile
maxDensities = readMaxScores(args.maxfile)

# printStore(storeDatabase)
# printStore(storeDesc)
dfAll, listdens = KDEAllZ(storeDatabase,storeDesc, args, maxDensities=maxDensities)
storeDatabase.close()
if storeDesc is not None:
	storeDesc.close()
dfAll.to_csv(args.outfile,index=False)
print("==========================================================================================", flush=True)
print("KDE for structures in ", fn, " are saved in ", args.outfile, " file", flush=True)
if len(listdens) >0:
	#print(listdens)
	dfmax=pd.DataFrame.from_dict(listdens)
	maxoutfile="maxKDE.csv"
	dfmax.to_csv(maxoutfile,index=False)
	print("Max KDE for each atomic number are saved in ", maxoutfile , " file", flush=True)
	
print("==========================================================================================", flush=True)



