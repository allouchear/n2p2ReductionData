'''
Read G symmetry functions from function.data (created by n2p2)
Create a hdf5 containing G functions by z value
'''

import pandas as pd
import numpy as np
import os
import sys
import argparse

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--infile", type=str, default="function.data", help="function.data file.")
	parser.add_argument("--outfile", type=str, default="functions.h5", help="output hdf file")
	args = parser.parse_args()
	return args

def getLinesFromInputFile(filepath):
	lines = []
	with open(filepath, 'r') as file:
		# read all lines in a list
		lines = file.readlines()

	if len(lines) == 0:
		print('Error : no data available at ', filepath)
		return None

	return lines

def getNextStructure(Z,dflist, lines, numLine, numStruct):
	nAtoms = int(lines[numLine].split()[0])
	numLine += 1
	for ia in range(nAtoms):
		an = int(lines[numLine].split()[0])
		row = [ numStruct,  an]
		row += lines[numLine].split()[1:]
		#print(drow)
		if an not in Z:
			Z += [an]
			dflist[an] = []
		dflist[an].append(row)
		numLine += 1

	numLine += 1 
	numStruct += 1
	return numLine, numStruct, Z, dflist
		
#Main

args = getArguments()

inFile = args.infile
outputFile = args.outfile

lines = getLinesFromInputFile(inFile)

print("# of lines = ",len(lines))
numLine=0
numStruct=1
Z = []
dflist = {}
nlinesAll=len(lines)
while numLine<nlinesAll:
	p=numLine/nlinesAll*100
	print('{:1s} = {:0.4f}\r'.format('\%',p),end='')
	numLine, numStruct, Z, dflist =getNextStructure(Z,dflist, lines, numLine, numStruct)

print(" ")
del lines
store = pd.HDFStore(outputFile,"w")
for z in Z:
	df = pd.DataFrame(dflist[z])
	del dflist[z]
	cols=["ID"]
	cols+=["Z"]
	for i in range(2,len(df.columns)):
		cols+=['G'+str(i-1)]
	df.columns=cols
	df.set_index('ID', inplace = True)
	cols = cols[1:]
	for col in cols:
		df[col] = df[col].astype(float)
	#print(df.info())
	#print(df)
	print("Z=",z, "Shape = ",df.shape)
	#print(list(df.columns))
	store.put('/Z_'+str(z), df)
	#print("End store")



store.close()
print("See ", outputFile, " file")

