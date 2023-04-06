import numpy as np
import pickle
import re
import sys
from itertools import tee
import pygeodesic.geodesic as geodesic 
from pandas import read_pickle
from .scanner import Scan
from .utils import align_point_cloud
from .utils import read_plyfile
from .utils import read_obj
from .utils import read_obj_mesh
from typing import List
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class Tape():
    
    def __init__(self, body_scan):
        self.body_scan = body_scan

    def pairwise(self, iterable):
        """ pairwise('ABCDEFG') --> AB BC CD DE EF FG """
        # adapted from https://docs.python.org/3/library/itertools.html#itertools.pairwise
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def linear_dist(self, locations:List):
        """ Straight line distance between points 
        Input: 
        locations - (list)landmark names (e.g 
                    [rt. acromion, rt. olecranon, rt. ulnar styloid])
        Output: 
        Cummulative distance between consecutive pairs.
        """
        # add 'ground' to the landmarks on the I body
        self.body_scan.landmarks['ground'] = self.body_scan.landmarks.get('ground', [-1,0,-1])
        
        dist = 0
        path=[np.array(self.body_scan.landmarks[locations[0]])]

        for (src, dest) in self.pairwise(locations):
            print('linear:',src,dest)

            src_coord = np.array(self.body_scan.landmarks[src])
            dest_coord = np.array(self.body_scan.landmarks[dest])
            if 'ground' in [src, dest]:
                dist += (src_coord[1] - dest_coord[1])
            else: 
                squared_dist = np.sum((src_coord - dest_coord)**2, axis=0)
                euclidean_dist = np.sqrt(squared_dist)
                dist += euclidean_dist
            path.append(dest_coord)

        return dist,np.array(path)
    
    def geodesic_dist(self, locations):  # Incomplete
        """Compute geodesic distance of landmark parallel to axis 
        Input: 
        pc- pointcloud
        axis- reference axis (0 - relative to the x axis, ...)
        landmark_name - like 'Sellion'
        """
        geoalg = geodesic.PyGeodesicAlgorithmExact(self.body_scan.vertices, self.body_scan.faces)
        path=[]
        distance=0
        for (src, dest) in self.pairwise(locations):
            print('geo',src,dest)
            src_coord = np.array(self.body_scan.landmarks[src])
            dest_coord = np.array(self.body_scan.landmarks[dest])
            src_index=np.argmin([np.linalg.norm(src_coord-p) for p in self.body_scan.vertices])
            dest_index=np.argmin([np.linalg.norm(dest_coord-p) for p in self.body_scan.vertices])
            _dist, _p = geoalg.geodesicDistance(dest_index,src_index)
            distance+=_dist
            path.append(_p)
        path=np.concatenate(path,axis=0)
        return distance,path

    def circumference_parallel_to_floor(self,location,segments=[],plot=False):
        """Compute circumfernce given by the convex hull
        of a cross section parrlel to floor starting at a given landmark 
        Input:         
        location - landmark_name - like 'Sellion'
        """
        point=self.body_scan.landmarks[location]
        nearest_point=np.argmin([ np.linalg.norm(point-p) for p in  self.body_scan.vertices])
        tape_width=7.9375/1000. #5/8 inch standard width of a tape measure
        y_sel=point[1]

        if segments  == []:
            selected_points=[(x,z) for i,(x,y,z) in enumerate(self.body_scan.vertices) if np.abs(y-y_sel) < tape_width]
            selected_points.append(self.body_scan.vertices[nearest_point,[0,2]])
            acluster=DBSCAN(eps=0.01,min_samples=5).fit(selected_points)
            selected_cluster=acluster.labels_[-1]
            perimeter_points=[(x,z) for i,(x,z) in enumerate(selected_points) if acluster.labels_[i]==selected_cluster]        
        else:
            perimeter_points=[(x,z) for i,(x,y,z) in enumerate(self.body_scan.vertices) if np.abs(y-y_sel) < tape_width and self.body_scan.point_labels[i] in segments]
        if plot:
            plt.scatter(*zip(*perimeter_points))
            plt.show()
        hull=ConvexHull(perimeter_points)
        path=[(perimeter_points[p][0],y_sel,perimeter_points[p][1]) for p in hull.vertices]
        path.append(path[0])
        return hull.area,np.array(path)
