from landmark.scanner import Scan
import polyscope as ps
import tensorflow as tf
import numpy as np
from landmark.measure import Tape
import polyscope as ps
import sys
from sral.fitsmpl import fit_smpl
import torch
#from ..models.wrapper import Model

scan_file=sys.argv[1]
lnd_file=scan_file.replace('.ply','.lnd')

body_scan=Scan(scan_file ,lnd_file,file_format='ply',forward_vector=(-1,-1,0),upward_vector=(0,0,1))
body_scan.clean()
#body_scan.align()
"""
# Test with pretrained model
model_dir=sys.argv[2]
lnd_names_file = sys.argv[3]
model = Model(model_arch="pointnet", 
                  landmark_names_file = lnd_names_file,
                  model_dir=model_dir, 
                  track_errors=True,
                  config=None)
                  
body_scan = Scan(pc_file, file_format='ply', lnd_file=None, pretrained_model=model)
tape = Tape(body_scan, pretrained_model=model)
"""
 
out=body_scan.fitsmplx()
body_scan.segment('./smplx_vert_segmentation.json')

ps.init()
ps.register_point_cloud('scan',body_scan.vertices)
ps.register_point_cloud('smpl_fit',out().vertices[0].detach().cpu().numpy())
ps.show()


### Test
tape = Tape(body_scan)
dist1,path1 = tape.linear_dist(['Rt. Lateral Malleolus', 'ground'])
dist2,path2 = tape.linear_dist(['Rt. Acromion', 'Rt. Olecranon', 'Rt. Ulnar Styloid'])
dist3,path3= tape.geodesic_dist(['Rt. Acromion', 'Rt. Olecranon','Rt. Ulnar Styloid'])
dist4,path4=tape.circumference_parallel_to_floor('Rt. Thelion/Bustpoint',segments=[10,9,12,11])

#dist3 = tape.geodesic_dist(['Rt. Thelion/Bust point', 'Lt. Thelion/Bust point'])
print(dist1, dist2,dist3, dist4)


ps.init()
ps.register_point_cloud('scan',body_scan.vertices)
ps.register_curve_network('path1',path1,'line')
ps.register_curve_network('path2',path2,'line')
ps.register_curve_network('path3',path3,'line')
ps.register_curve_network('path4',path4,'line')
ps.show()

   
