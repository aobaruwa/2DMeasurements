import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import numpy as np
import xml.etree.ElementTree as ET
import pdb
from scipy.spatial.transform import Rotation as R

def read_plyfile(fname):
    plydata = PlyData.read(fname)
    vertices = np.array([plydata['vertex']['x'], 
                        plydata['vertex']['y'],
                        plydata['vertex']['z']]).T
    faces = np.array([f[0] for f in plydata["face"].data])
    return vertices, faces

def read_lnd(fname):
    lnd_data=open(fname,'r').readlines()
    start=False
    landmark_dict={}
    for l in lnd_data:
#        print(l,start,len(lnd_data))

        if l.startswith("AUX ="):
            start=True
            continue
        if l.startswith("END ="):

#            print("Break")
            break
        if start:            
            data=l.split()
            measurment_number=data[0]
            # No documentation on what thes 3 numbers are but d1 and d2 are -999 if point is missing
            d1=data[1]
            d2=data[2]
            d3=data[3]
            x=float(data[4])/1000.
            y=float(data[5])/1000.
            z=float(data[6])/1000.
            if d2 == "-999": continue
            name=" ".join(data[7:])
            landmark_dict[name]=(x,y,z)
    return landmark_dict
            

def write_ply(data,fname):
    vertex=np.array( [(x,y,z) for x,y,z in data],dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(fname)

def read_obj_mesh(fname):
    vertices = []
    faces = []
    for line in open(fname,'rb').readlines():
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                if len(face) > 3: # triangulate the n-polyonal face, n>3
                    faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                else:
                    faces.append([face[j][0]-1 for j in range(len(face))])
            else: pass
    return np.array(vertices), np.array(faces)

def read_obj(fname):
    data=open(fname).readlines()
    pcloud=[]
    for l in data[3:]:
        if not l.startswith("v "):continue
        ptype,x,y,z=l.split()
        pcloud.append(np.array([[float(x),float(y),float(z.strip('\n'))]]))
    pcloud=np.concatenate(pcloud)
    return pcloud

def rotate_scan_upright(pc):
    #Rottate the longest axis into the Z direction
    x_diff=np.max(pc[:,0])-np.min(pc[:,0])
    y_diff=np.max(pc[:,1])-np.min(pc[:,1])
    z_diff=np.max(pc[:,2])-np.min(pc[:,2])
    max_axis=np.argmax((x_diff,y_diff,z_diff))
    if max_axis == 2: #No reason to rotate z is already the long axis
        rotational_offsets=[(0,0,0)]
    elif max_axis == 0: #X is the long axis rotate around y
        rotational_offsets=[(0,90,0)]
    elif max_axis == 1: #Y is long axis rotate around x
        rotational_offsets=[(90,0,0)]
    rotational_offsets.append((0,0,-45)) #Rotate 45 degrees to match ceasar data
    new_pc=pc.copy()
    for r in rotational_offsets:
        rotation=R.from_euler('xyz',r,degrees=True)
        new_pc=rotation.apply(new_pc)
    return new_pc,rotational_offsets

def center_point_cloud(pc):
    # Move the feet to the floor and center body on z-axis
    # Scan must already be up-right with the head in the postive z-direction 
    x_offset= -1*np.mean(pc[:,0])
    y_offset=-1*np.mean(pc[:,1])
    z_offset=-1*np.min(pc[:,2]) -1 #floor in cesar data is -1

    offsets=(x_offset,y_offset,z_offset)
    new_pc=pc.copy() + offsets

    return new_pc,offsets

def align_point_cloud(pc):
    new_pc,rotational_offsets=rotate_scan_upright(pc)
    new_pc,translational_offsets=center_point_cloud(new_pc)
    return new_pc, [translational_offsets,rotational_offsets]

    

def sample_pc(data,size=200):
    if len(data) >= size:
        indexes=np.random.choice( range(len(data)),size,replace=False )
    else:
        indexes=np.random.choice( range(len(data)),size,replace=True )
        
    return data[indexes]


def plot(data,colors=[]):
    if type(data) != list: data=[data]
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xlim3d(-1,1)
    ax1.set_ylim3d(-1,1)
    ax1.set_zlim3d(-1,1)
    for ax in [ax2,ax3,ax4]:
        ax.set_ylim((-1,1))
        ax.set_xlim((-1,1))
    
    if colors==[]:
        for d in data:
#           x,y,z=zip(*rotate(d))
            x,y,z=zip(*d)
            ax1.scatter(x,y,z,marker='o',s=.6)
            ax2.scatter(x,y,marker='o',s=.6)
            ax3.scatter(x,z,marker='o',s=.6)
            ax4.scatter(y,z,marker='o',s=.6)
    else:
        start=True
        for d,cs in zip(data,colors):
#            x,y,z=zip(*rotate(d))
            x,y,z=zip(*d)
            if start:
                alpha=1
                cmap='plasma'
                start=False
            else:
                alpha=0.2
                cmap='gist_gray'
            
            
            ax1.scatter(x,y,z,marker='o',s=.3,c=cs, vmax=np.percentile(cs,99.99) ,vmin=np.percentile(cs,1),cmap=cmap ,alpha=alpha)
            ax2.scatter(x,y,marker='o',s=.6,c=cs, vmax=np.percentile(cs,99.99) ,vmin=np.percentile(cs,1),cmap=cmap,alpha=alpha)
            ax3.scatter(x,z,marker='o',s=.6,c=cs, vmax=np.percentile(cs,99.99) ,vmin=np.percentile(cs,1),cmap=cmap,alpha=alpha)
            ax4.scatter(y,z,marker='o',s=.6,c=cs, vmax=np.percentile(cs,99.99) ,vmin=np.percentile(cs,1),cmap=cmap,alpha=alpha)
        
    plt.show()





def plot_landmark(data,landmarks):
    if type(data) != list: data=[data]
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xlim3d(-1,1)
    ax1.set_ylim3d(-1,1)
    ax1.set_zlim3d(-1,1)
    for ax in [ax2,ax3,ax4]:
        ax.set_ylim((-1,1))
        ax.set_xlim((-1,1))
     
       
    for d in data:
        #x,y,z=zip(*rotate(d))
        x,y,z=zip(*d)
        ax1.scatter(x,y,z,marker='o',s=.6)
        ax2.scatter(x,y,marker='o',s=.6)
        ax3.scatter(x,z,marker='o',s=.6)
        ax4.scatter(y,z,marker='o',s=.6)
    lms=[]
    for lm,point in landmarks.items():
        lms.append(point)
    x,y,z=zip(*lms)
    ax1.scatter(x,y,z,marker='x',s=1,color='r')
    ax2.scatter(x,y,marker='x',s=1,color='r')
    ax3.scatter(x,z,marker='x',s=1,color='r')
    ax4.scatter(y,z,marker='x',s=1,color='r')
    
    plt.show()
    

def read_landmark_xml(xml_file):
    landmarks={}
    tree = ET.parse(xml_file)
    root=tree.getroot()
    for lm in root.find('scan').findall('landmark'):
        name=lm.get('name')
        coords=[float(lm.get(i)) for i in ['x','y','z']]
        landmarks[name]=coords
    return landmarks


def write_landmark_xml(out_file,landmarks):
# A function that takes a dictionary of landmarks
# Each key should be the name a landmark and each value a tuple of x,y,z coords  
    root=ET.Element('landmarks')
    scan=ET.SubElement(root,'scan')
    scan.set('path','./')
    for name,(x,y,z) in landmarks.items():

        lm=ET.SubElement(scan,'landmark')
        lm.set('name',name)
        lm.set('id',name)
        lm.set('scan','')
        lm.set('x',str(x))
        lm.set('y',str(y))
        lm.set('z',str(z))
    tree=ET.ElementTree(root)
    tree.write(out_file)

    
if __name__=="__main__":
    data=read_plyfile('/home/jsearcy/University Of Oregon/CAESAR Data AE2000/North America/PLY and LND NA/csr0079b.ply')
    data2=read_obj('/home/jsearcy/UO Data Science Dropbox/Private Datasets/Design/Susan/Size 16 by Chest/Height Group 0/002-20170420-006 0/Standard.obj')
    sample=sample_pc(data,size=1000)
    plot(sample)
    sample2=sample_pc(data2,size=1000)
    plot(sample2)
