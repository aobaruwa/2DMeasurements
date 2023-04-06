from glob import glob
import pickle
import os 
import numpy as np
import tensorflow as tf
from landmark import utils
import sys


#quick script to plot an obj file and landmark data args are the file name to the obj and the file name to the xml

fname=sys.argv[1] #'/gpfs/projects/datascience/shared/OtherScans/3DMDScan.obj'
landmark_xml=sys.argv[2] #'ml_pred_FL071819F04_B1_LINED.xml'

if fname.endswith('.obj'):
    pc=utils.read_obj(fname)
    pc,offsets=utils.align_point_cloud(pc)
if fname.endswith('.ply'):
    pc=utils.read_plyfile(fname)

landmarks=utils.read_landmark_xml(landmark_xml)

xs=[pc]

utils.plot_landmark(pc,landmarks)

