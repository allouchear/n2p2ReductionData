import os
import sys
import numpy as np
import pandas as pd
import argparse

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--infile', default='resultsKDE.csv', type=str, help='')
	parser.add_argument('--outfile', default='numStructs.csv', type=str, help='')
	parser.add_argument("--method", type=str, default="Regular", help="method : Regular, Logarithmic, Smallest")
	parser.add_argument("--p", type=float, default=10.0, help=" real : % of selected structures, default=10.0")

	args = parser.parse_args()

	return args

def selRegular(dfkde):
	dfkde.sort_values(by=['Dens'], inplace=True, ascending=True)
	dfkde.reset_index(inplace=True)
	#print(dfkde)
	ns=dfkde.shape[0]
	nsel=int(ns*args.p/100.0)
	nstep=ns//nsel
	print("Total number of structures=",ns)
	print("nStep=",nstep)
	listi=list(range(0,ns))
	listi=listi[0::nstep]
	print("Number of selected structures=",len(listi))
	#print(len(listi))
	#print(listi)
	dfkde = dfkde.iloc[listi]
	#print(dfkde)
	return dfkde

def selSmallest(dfkde):
	dfkde.sort_values(by=['Dens'], inplace=True, ascending=True)
	dfkde.reset_index(inplace=True)
	#print(dfkde)
	ns=dfkde.shape[0]
	nsel=int(ns*args.p/100.0)
	print("Total number of structures=",ns)
	listi=list(range(0,nsel))
	print("Number of selected structures=",len(listi))
	#print(len(listi))
	#print(listi)
	dfkde = dfkde.iloc[listi]
	#print(dfkde)
	return dfkde

def selLog(dfkde):
	dfkde.sort_values(by=['Dens'], inplace=True, ascending=True)
	dfkde.reset_index(inplace=True)
	#print(dfkde)
	ns=dfkde.shape[0]
	nsel=int(ns*args.p/100.0)
	print("Total number of structures=",ns)
	n=0
	nss = nsel
	while n<nsel:
		arr = np.logspace(0, np.log(ns)/np.log(10), num=nss, endpoint=False)
		arr = np.array(arr,dtype=int)
		arru = np.unique(arr)
		nd=arr.shape[0]-arru.shape[0]
		#print(arru)
		#print(arru.shape)
		nss+=nd
		n=arru.shape[0]
	listi=list(arru)
	#print(len(listi))
	#print(listi)
	print("Number of selected structures=",len(listi))
	dfkde = dfkde.iloc[listi]
	#print(dfkde)
	return dfkde



args = getArguments()
infile=args.infile 
outfile=args.outfile 
dfkde=pd.read_csv(infile)
if args.method.upper()=='REGULAR':
	print("Method = ",  args.method)
	dfkde=selRegular(dfkde)
elif args.method.upper()=='SMALLEST':
	print("Method = ",  args.method)
	dfkde=selSmallest(dfkde)
elif args.method.upper()=='LOGARITMIC':
	print("Method = ",  args.method)
	dfkde=selLog(dfkde)
else:
	print("==================================")
	print("Error : unknown method ", args.method,flush=True)
	print("==================================")

dfkde.sort_values(by=['ID'], inplace=True, ascending=True)
dfkde['0']=dfkde['ID']-1
dfkde=dfkde['0']
dfkde.to_csv(outfile,index=False)

print("number of selected structures are saved in ", outfile, " file", flush=True)


print("See ", outfile, ' file')

