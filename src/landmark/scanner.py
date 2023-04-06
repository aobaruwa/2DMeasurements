from .utils import read_plyfile, read_obj,read_landmark_xml,read_lnd
import os
import numpy as np
import pickle
import re
from scipy.spatial.transform import Rotation
import pdb
from sral.fitsmpl import fit_smpl
import json
from scipy.spatial import KDTree
from tqdm import tqdm


try:
    import open3d as o3d
    use_o3d=True
except:
    use_o3d=False










    
class Scan():
    def __init__(self, pc_file, lnd_file=None, file_format='pk', 
                 pretrained_model=None,forward_vector=None,upward_vector=None):
        self.pc_file = pc_file
        self.lnd_file = lnd_file
        self.file_format= file_format
        self.vertices = None
        self.landmarks = None
        self.landmark_names=None
        self.faces = None

        self.alignment_vector=np.array([forward_vector,upward_vector])
        self.offset=None
        self.read(pretrained_model)
        self.in_mm=False
        
    def align(self):
        target_vector=np.array([(0,0,1),(0,1,0)])
        
        self.rot,__=Rotation.align_vectors(target_vector,self.alignment_vector)
        
        self.vertices=self.rot.apply(self.vertices)
        self.offset=(0,np.min(self.vertices[:,1]),0)
        self.vertices=self.vertices-self.offset
        for l in self.landmarks:
            self.landmarks[l]=self.rot.apply(self.landmarks[l])-self.offset


        
    def fitsmplx(self,verbose=False):
        import torch
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.align()
        if len(self.vertices) > 10000:
            fit_data=torch.Tensor(self.vertices[np.random.choice(len(self.vertices),10000,replace=False)])
        else:
            fit_data=torch.Tensor(self.vertices)
        
        self.smpl_fit=fit_smpl(torch.unsqueeze(fit_data,dim=0).to(device),verbose=verbose)

        return self.smpl_fit
        

        

    def clean(self):
        point_vals={ int(i):0 for i in range(0,len(self.vertices))  }
        all_faces={int(i):-i for i in self.faces.flatten()}

        new_points=[]
        new_index=0
        for pi in point_vals:
            if pi in all_faces:
                new_points.append(self.vertices[pi])
                point_vals[pi]=new_index
                new_index+=1
            else:
                pass
        self.vertices=np.array(new_points)
        for index,(p1,p2,p3) in enumerate(self.faces):
            self.faces[index]=np.array([point_vals[p1],point_vals[p2],point_vals[p3]])
            
        
    def normalize_names(self, landmark_names):
        # remove names with forward slash and pound symbols
        norm_names = []
        for name in landmark_names:
            norm = re.sub(r"[/#]", "", name.lower()).replace(" ", "_") 
            norm_names.append(norm)
        return norm_names

    def read(self, pretrained_model):
        """Read 3D body scan file, vertices and landmarks"""
        
        print(f"\nReading {self.file_format} file . . .\n")
        
        #else:
        if self.file_format == 'pk':  ### self.faces ??????????
            pc_obj = pickle.load(open(self.pc_file, "rb"))
            key =list(pc_obj.keys())[0]
            self.landmarks = {}
            self.vertices = []
            self.landmarks_names = list(self.landmarks.keys())
            self.vertices = pc_obj[key]['PC']   
               
        elif self.file_format in ['ply','obj'] and use_o3d:
            self.mesh = o3d.io.read_triangle_mesh(self.pc_file)
            self.vertices=np.asarray(self.mesh.vertices)
            if np.max(self.vertices) > 100:
                self.vertices/=1000. #probably in mm
                self.in_mm=True
            self.faces=np.asarray(self.mesh.triangles)      
        elif self.file_format == 'ply' and not use_o3d:
            self.vertices,self.faces=read_plyfile(self.pc_file)
        else:
            raise ValueError('Unknown point cloud file format, format must be \
                            (pk, ply, or obj)')
        if pretrained_model: 
            self.landmarks, self.landmark_names = pretrained_model._predict(self.vertices)
        elif self.lnd_file:
            self.add_landmarks(self.lnd_file)
        
    def add_landmarks(self,landmark_file,tag=''):
        """Read a landmark file and add each landmark with an optional tag"""
        if landmark_file.endswith('.lnd'):
            lnd_dict=read_lnd(landmark_file)
        elif landmark_file.endswith('.xml'):
            lnd_dict=read_landmark_xml(landmark_file)
        else:
            raise ValueError('Unknown point cloud file format, format must be \
                            (xml or lnd)')            
        if self.landmarks == None:
            self.landmarks={}
            self.landmark_names=[]

        for key in lnd_dict:
            name=tag+key
            if name in self.landmarks:
                raise ValueError('Cannot Add Landmark,'+ name+' already exists' )
            else:
                self.landmarks[name]=lnd_dict[key]
                self.landmark_names.append(name)

    def segment(self,seg_file):
        segments=json.loads(open(seg_file).read())
        self.bp_cat=list(segments.keys())
        self.bp_cat.append('border')
        lookup={}
        for s in segments:
            if s in ['eyeballs']:continue
            for p in segments[s]:
                if p in lookup: lookup[p]=self.bp_cat.index('border')
                else:
                    lookup[p]=self.bp_cat.index(s)

        for i in [8811, 8812, 8813, 8814, 9161, 9165]:
            lookup[i]=self.bp_cat.index('neck')

        k,v=zip(*lookup.items())
        labels=np.array(v)[np.argsort(k)]
        nn=self.get_nn(self.vertices,self.smpl_fit().vertices[0].detach().cpu().numpy())
        lookup[-1]=-1
        self.lookup=lookup
        self.point_labels=np.array([lookup[n] for n in nn])


    def get_nn(self,pc1,pc2,dist=.03):
        k1=KDTree(pc1)
        k2=KDTree(pc2)
        print('Query Tree')
        near_points = k1.query_ball_tree(k2, r=0.2)
        nn=[]
        for i,points in tqdm(enumerate(near_points)):
            if len(points) >0:
                min_point=np.argmin(np.linalg.norm(pc2[points]-pc1[i,:],axis=-1))
                nn.append(points[min_point])
            else:
                nn.append(-1)
        return nn

