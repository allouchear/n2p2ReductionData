import os
import sys
import numpy as np
import pandas as pd
import argparse

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--infile', default='input.data', type=str, help='')
	parser.add_argument('--numfile', default='numStructs.csv', type=str, help='')
	parser.add_argument('--outfile', default='selInput.data', type=str, help='') 	

	args = parser.parse_args()

	return args

def buildFile(infile, outfile, numfile):
	dfnum=pd.read_csv(numfile)
	fin=open(infile,"r")
	fout=open(outfile,"w")
	natoms=0
	numS = 0;
	lines = fin.readlines()
	#print(dfnum)
	#print(dfnum['0'])
	listN=dfnum['0'].to_list()
	#print(listN)
	sel=False
	for line in lines:
		if line.find('begin')!= -1:
			numS += 1
			sel=numS in listN
			#print("numS=",numS, 'sel=',sel)
		if sel:
			fout.write(line)
	fin.close()
	fout.close()

args = getArguments()
infile=args.infile 
outfile=args.outfile 
numfile=args.numfile 
buildFile(infile, outfile, numfile)
print("See ", outfile, ' file')

