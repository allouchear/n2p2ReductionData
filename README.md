n2p2ReductionData - some script to reduce structure database using clustering approaches
=========================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository provides severl script (python + bash) to reduce the database of structures used to traing NN network potential with n2p2 package

# buiildGh5.py
 build a .h5 file using function.data contening G function produced by nnp-scaling program of n2p2
 Input  file : function.data (default), required. You can used --infile=otherfile to change the name of input file
 Output file : functions.h5 (default). You can change the name using --outfile=otherfile to change the name of output file

# Clustering.py
 Search list of selected structures based KMeans, DBSCAN or HDBSCAN clustering methods
 Input  file : functions.h5 (default), required. You can used --infile=otherfile to change the name of input file
 Outut  file : numStructs.csv (default), required. You can used --outfile=otherfile to change the name of output file
 The default clustering approach is KMeans. You can change it using --method=DBSCAN or --method=HDBSCAN
 The default minsample hperparameter is 2. You can change it by --minsample=OtherValue
 The percentage of selected structures by cluster is set to 0.20%. TO change it : --p=newValue
 The eps value of eps for DBSCAN is computed using  (NearestNeighbors+Knee) method. You can change it using --eps=newPositiveValue

# buildSelectedData.py
 build a selInput.data file using input.data (the database for nnp-train) and  numStructs.csv
 Input  file : input.data (default), required. You can used --infile=otherfile to change the name of input file
 Input  file :  numStructs.csv (default), required. You can used --numfile=othernumfile to change the name of numfile
 Outut  file : selInput.data (default), required. You can used --outfile=otherfile to change the name of output file

## Build the reduced data
To recudce the database, you have to run, in this order :
```
buiildGh5.py
Clustering.py
buildSelectedData.py
```
As an example, see xAllKMeans, xAllDBSCAN and xALLHDBSCAN in testH2O folder 

# Authors
 - Abdulrahman Allouche (Lyon 1 University)

# License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).
