from dgl.dataloading import GraphDataLoader
from data import PCDataset
from model import GCN, GAT
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import os

BATCH_SIZE = 10
N_EPOCHS = 100

TITLE = "GAT_10_EPOCHS"
# load data
dataset = PCDataset()
num_labels = dataset.num_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create dataloaders
dataloader = GraphDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

h_dims = [16,16]
model = GCN(in_dim=3, h_dims=h_dims, n_classes=3*75)

model = model.to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters())

training_loss = []
writer = SummaryWriter(TITLE)
# training
for epoch in range(N_EPOCHS):
    for i, (batched_graph, labels) in enumerate(dataloader):

        batched_graph = batched_graph.to(device)
        feats = batched_graph.ndata['feat']
        labels = labels.to(device)
        preds = model(batched_graph, feats)
        
        loss = loss_fn(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        writer.add_scalar('train_loss', loss.item(), i)
    print("preds:",preds[0], "labels:", labels[0])
    print("epoch loss:",loss.item())
data_dir = "/projects/datascience/shared/DATA/graph_data"
torch.save(model.state_dict(),os.path.join(data_dir, TITLE+"model.pt"))
