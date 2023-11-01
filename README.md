n2p2ReductionData - some scripts to reduce structure database using several machine learning approaches
=======================================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository provides several script (python + bash) to reduce the database of structures used to train NN network potential with n2p2 package.

## buildGh5.py
 build a .h5 file using function.data contenning G function produced by nnp-scaling program of n2p2 
 - Input  file : function.data (default), required. You can used --infile=otherfile to change the name of input file
 - Output file : functions.h5 (default). You can change the name using --outfile=otherfile to change the name of output file

## Clustering.py
 Search list of selected structures based on KMeans, DBSCAN or HDBSCAN clustering method.
 - Input  file : functions.h5 (default), required. You can used --infile=otherfile to change the name of input file
 - Outut  file : numStructs.csv (default). You can used --outfile=otherfile to change the name of output file
 - The default clustering approach is KMeans. You can change it using --method=DBSCAN or --method=HDBSCAN
 - The default minsample hperparameter is 2. You can change it by --minsample=OtherValue
 - The percentage of selected structures by cluster is set to 0.20%. To change it : --p=newValue. Please note that for DBSCAN and HDBSCAN, all outliers structures are selected.
 - The optimal value of eps hyperparameter of DBSCAN is computed using NearestNeighbors+Knee method. You can change it using --eps=FixedPositiveValue
 - By default no reduction of dimension. If needed, add: --reddim=PCA
 - To define the dimension after reduction, set kr. --kr=integer or a real real between 0 and 1.0 (see scikitlearn documentaion for PCA)
 - By defeault the data are not scaled. To do it, use : --scaling=MinMax, Standard or MaxAbs
 - By default all data (rows) are used. To reduce data, use --reddata=MaxG or --reddata=StdG. In this case, We search the G column with max (MaxG or StdG) value. The data are sorted using this column and the data are reduced to kdeddat, taking rows with linear step.
 - Set the data size using --kreddata=integervaluer
   
## SelectionOnGrid.py
 Search list of selected structures based on G values on grid.
 - Input  file : functions.h5 (default), required. You can used --infile=otherfile to change the name of input file
 - Outut  file : numStructs.csv (default). You can used --outfile=otherfile to change the name of output file
 - The default method to make a reduction of dimensions is PCA. You can change it using --method=TSNE or --method=None (without reduction)
 - By default no scaling on G. You can change it by --scaling=Standard, --scaling=MinMax, --scaling=AbsMax, or --scaling=None (default)
 - The number of dimensions after reduction is 1. To change it, use --k=value, where value represents the number of dimensions for t-SNE (from 1 to 3) or PCA. For PCA, k can be a real number between 0 and 1.0. In this case, the number of dimensions is computed automatically based on the amount of variance that needs to be explained, which is greater than the percentage specified by n_components (see scikit-learn documentation)
 - The percentage used to select number of grid points is set to 0.20%. To change it : --p=newValue.  the number of grid points m = int((number of dataset/100*percentage)**(1.0/n_components). If p<0 : m=int(-p) for each direction

## buildSelectedData.py
 build a selInput.data file using input.data (the database for nnp-train) and  numStructs.csv
 - Input  file : input.data (default), required. You can used --infile=otherfile to change the name of input file
 - Input  file :  numStructs.csv (default), required. You can used --numfile=othernumfile to change the name of numfile
 - Outut  file : selInput.data (default). You can use --outfile=otherfile to change the name of output file

## Build the reduced data
To reduce the database, you have to run, in this order :
```
python buildGh5.py
python Clustering.py
# or python SelectionOnGrid.py
python buildSelectedData.py
```
As an example, see xAllKMeans, xAllDBSCAN, xAllSelOnGrid and xALLHDBSCAN in testH2O folder 

# Authors
 - Abdulrahman Allouche (Lyon 1 University)

# License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).
