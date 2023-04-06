# Landmark
A machine learning algorithm to landmark 3d body scans

# Talapas Enviroment
```
conda create --name env_landmark --clone anaconda-tensorflow2-gpu-20210607-nvidiatest
conda activate env_landmark
pip install matplotlib
pip install plyfile
pip install scipy
```
everytime before working
```
conda activate env_landmark
```


# Workflow
  1. The *lmcreateDS.py <directory>* scripts takes as input a directory containing the CAESAR data. It processes in parrelel each ply and lnd file pair into python pickle file containing a dictionary of the x,y,z point clouds under the key *PC* and a dictionary of Landmarks unders the key *LM*. These pickle files are used for all the down stream model training and are stored in the data/ directory of the package.
  
  2. train_classifier.py will load all *.pk files and great a numpy array for the point clouds and the target land_mark (hard coded at the momment). This script will train a model for 50 epochs and then save the trained weights
  
  
  
