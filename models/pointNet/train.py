from torch.optim import Adam
from data import PcDataset
from old_point_net import PointNet
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

class Trainer():
    """Main class that handles the logic for training a pointnet model
       defined in model.py
    """
    def __init__(self, net, n_epochs, grad_accum_steps, optimizer, writer):
        self.net = net
        self.n_epochs = n_epochs
        self.grad_acum_steps = grad_accum_steps
        self.optimizer = optimizer
        self.writer = writer
        self.current_epoch=0
        
    def train_step(self, train_loader):
        for step, (data, labels, label_masks) in enumerate(train_loader):
            self.optimizer.zero_grad()
            data, labels, label_masks = data.to(device), labels.to(device), label_masks.to(device)
            logits, inp_transform, feat_transform  = self.net(data)
            loss = self.loss_fn(logits.float(), labels.float(), inp_transform.float(), feat_transform.float(), label_masks.float())
            loss.backward()
            if (step+1)% self.grad_acum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
    
        ## calculate and save epoch loss
        print('train_loss', loss.item())

        self.writer.add_scalar('train_loss', loss.item(),  self.current_epoch)
        return loss.item()

    def val_step(self, val_loader):
        step_loss = []
        for step, (data, labels, label_masks) in enumerate(val_loader):
            data, labels, label_masks = data.to(device), labels.to(device), label_masks.to(device)
            logits, inp_transform, feat_transform  = self.net(data)
            loss = self.loss_fn(logits.float(), labels.float(), inp_transform.float(), feat_transform.float(), label_masks.float())
            step_loss.append(loss.item())
        
        avg_loss = np.mean(step_loss)
        print('val_loss', avg_loss)
        self.writer.add_scalar('val_loss', avg_loss,  self.current_epoch)

        return avg_loss

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for e in range(self.n_epochs):
            # train step
            self.train_step(train_loader)
            # eval step
            with torch.no_grad():
                val_loss = self.val_step(val_loader)
        
            if val_loss < best_val_loss:
                print(f"{val_loss} < {best_val_loss} saving model ...")
                model_path = os.path.join(os.path.abspath(""), "model.pt")
                torch.save(net, model_path)
                best_val_loss = val_loss
            self.current_epoch+=1
        return best_val_loss

    def loss_fn(self, logits, labels, m3x3, m64x64, masks, alpha=0.0001):
        logits = logits.view(-1,74,3)
        labels = labels.view(-1,74,3)
        masks = masks.view(-1,74,1)

        masked_logits  = torch.mul(logits, masks)
        masked_labels = torch.mul(labels, masks)
        bs = masked_logits.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        if logits.is_cuda:
            id3x3 = id3x3.cuda()
            id64x64 = id64x64.cuda()
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        mse_loss = F.mse_loss(masked_logits, masked_labels)
        ortho_loss =  alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)
        print("ortho_loss", ortho_loss, "mse_loss", mse_loss)
        return mse_loss + ortho_loss

if __name__=='__main__':
    n_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    num_points = int(sys.argv[3])
    train_file = sys.argv[4]
    val_file = sys.argv[5]
    log_dir = sys.argv[6]
    grad_accum_steps = int(sys.argv[7])


    # Data Loaders 
    train_set = PcDataset(train_file, num_points)
    val_set = PcDataset(val_file, num_points)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=4, drop_last=True)

    # Training Ops
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} . . .")

    net = PointNet(classes=222) #Model(num_points, dropout_rate=0.7, K1=3, K2=64)
    net.to(device)
    print(net)  
    
    optimizer = Adam(net.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(net, n_epochs, grad_accum_steps, optimizer, writer)
    trainer.train(train_loader, val_loader)
    # save model
    model_path = os.path.join(os.path.abspath(""), "model.pt")
    torch.save(net, model_path)
    
