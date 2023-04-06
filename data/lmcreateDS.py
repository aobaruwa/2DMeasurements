import os
from glob import glob
import multiprocessing
import atexit
import sys
import pickle
from landmark import utils
import numpy as np
this_dir=os.path.dirname(__file__)
example_size=10000


output_directory=os.path.join(this_dir,'data')

def get_subject_id_from_fname(fname):
    basename=os.path.basename(fname)
    if (".ply" not in basename) and  (".lnd" not in basename):
        raise NameError("Input file is not a ply or landmark file!")
    idvalue= basename[:basename.find(".")]
    return idvalue
    
def get_matches(plys,lnds):
    ply_dict={}
    for i,f in enumerate(plys):
        _id=get_subject_id_from_fname(f)
        ply_dict[_id]=i

    lnd_dict={}
    for i,f in enumerate(lnds):
        _id=get_subject_id_from_fname(f)
        lnd_dict[_id]=i
    
    file_matches=[]
    for f in lnd_dict:
        if f in ply_dict:
            ply_index=ply_dict[f]
            lnd_index=lnd_dict[f]
            file_matches.append( (plys[ply_index],lnds[lnd_index])  )
        else:
            print('No match for file',f)

    return file_matches

def write_files(in_files):
    ply_file,lnd_file=in_files
    s1=get_subject_id_from_fname(ply_file)
    s2=get_subject_id_from_fname(lnd_file)
    assert s1==s2
    
    data_dict={s1:{}}

    ply_file=utils.read_plyfile(ply_file)
    lnd_dict=utils.read_lnd(lnd_file)
    if np.max(ply_file)  > 100: #This is a file in mm instead of meters about 25% of the Ceasar scans
        ply_file=ply_file/1000.
    data_dict[s1]['PC']=ply_file
    data_dict[s1]['LM']=lnd_dict
    
    outfile=os.path.join(output_directory,s1+".pk")
    pickle.dump(data_dict,open(outfile,'wb'))
    return


if __name__=='__main__':

    input_directory=sys.argv[1]
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    ply_string=os.path.join(input_directory,'**','*.ply')
    lnd_string=os.path.join(input_directory,'**','*.lnd')
    
    ply_files=glob(ply_string,recursive=True)
    lnd_files=glob(lnd_string,recursive=True)

    matched_files=get_matches(ply_files,lnd_files)

    print('Found ',len(ply_files),' PLY files')
    print('Found ',len(lnd_files),' LND Files')
    print('Found ',len(matched_files),' Matches')

#    test=write_files(matched_files[0])

    p=multiprocessing.Pool(28)
    atexit.register(p.close)

    p.map(write_files,matched_files)
    

