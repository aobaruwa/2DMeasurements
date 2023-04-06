from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from glob import glob
from tqdm import tqdm
from utils import read_lnd
import dgl
import os
import multiprocessing as mp
import numpy as np
import torch
import trimesh as tm
import networkx as nx
import torch.nn.functional as F
print("done importing !")
data_dir = "/projects/datascience/shared/CAESAR Data AE2000/North America/PLY and LND NA"
save_dir = "/projects/datascience/shared/DATA/graph_data"

class PCDataset(DGLDataset):
    def __init__(self, data_dir=data_dir, save_dir=save_dir, save_graph=True, verbose=False):
        self.data_dir = data_dir
        save_graph=save_graph
        super(PCDataset, self).__init__(name="PCDataset",
                                        save_dir=save_dir,
                                        verbose=verbose)

    def process(self):
        self.ply_files = glob(os.path.join(self.data_dir, "*.ply"))
        self.lnd_files = glob(os.path.join(self.data_dir, "*.lnd"))
        plyfiles = []; self.labels = []
        
        for ply_file in self.ply_files:
            lnd_file = os.path.join(self.data_dir, os.path.basename(ply_file)[:-4]+'.lnd')
            if os.path.exists(lnd_file):
                labels = read_lnd(lnd_file)
                plyfiles.append(ply_file)
                self.labels.append(labels)
        print("dataset_size: {}".format(len(plyfiles)))
        
        pool = mp.Pool()
        L = mp.Manager().list()
        for fname in plyfiles:
            tqdm(pool.apply_async(self.create_graph, args = [fname, L]))
        pool.close()
        pool.join()
        self.graphs = list(L)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print (self.graphs)
        return 

    def create_graph(self, ply_file, L):
        mesh = tm.load_mesh(ply_file)
        graph = mesh.vertex_adjacency_graph
        adj_matrix = nx.to_scipy_sparse_matrix(graph, dtype=np.uint8)
        graph =  dgl.from_scipy(adj_matrix, device=torch.device('cpu'))
        graph.ndata['feat'] = torch.tensor(mesh.vertices, dtype=torch.float32)
        L.append(graph)
        return graph

    def save(self):
        #save the graph list and the labels
        graph_path = os.path.join(self.save_dir, 'dgl_graph.bin')
        print("writing graphs to memory . . .")
        save_graphs(str(graph_path), self.graphs, 
                    {'labels': self.labels})

    def load(self):
        graph_path = os.path.join(self.save_dir, 'dgl_graph.bin')
        graphs, label_dict = load_graphs(str(graph_path))
        self.graphs = graphs
        self.labels = label_dict['labels']
        print("done loading data from cache . . .")
        
    def has_cache(self):
        graph_path = os.path.join(self.save_dir, 'dgl_graph.bin')
        return os.path.exists(graph_path)
  
    @property
    def num_labels(self):
        """ Number of labels for each graph, i.e. number of prediction tasks. """
        return 3*75

    def __getitem__(self, idx):
        """ Get graph and label by index"""
        if self._transform is None:
            g = self.graphs[idx]
        else:
            g = self._transform(self.graphs[idx])
        return g, self.labels[idx]
    
    def __len__(self):
        """ Number of graphs in the dataset. """
        return len(self.graphs)
