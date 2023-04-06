import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import PcDataset
from train import Trainer
import sys 

ACTIVATION = {'relu': nn.ReLU, 'elu':nn.ELU, 'gelu': nn.GELU}
class Tnet(nn.Module):
    def __init__(self, trial, activation_fn, in_dim, k):
        super().__init__()
        self.trial = trial
        self.in_dim = in_dim
        self.k = k
        self.activation_fn = activation_fn
        n_layers = self.trial.suggest_int("n_layers", 3, 8, step=1)
        activation_fn = self.trial.suggest_categorical("relu", "elu", "gelu")
        cnn_dim_range = [3, 16, 32, 64, 128, 256, 512, 1024, 2048]
        fc_dim_range = [2048, 1024, 512, 256]
        self.layers = []

        in_features = self.in_dim
        ## Conv layers
        for i in range(n_layers):
            ## pick a higher dim from the list
            pos = cnn_dim_range.index(in_features)
            out_features = self.trial.suggest_categorical(f"out_channels{i}", cnn_dim_range[pos+1:]) 
            self.layers.append(nn.Conv1d(in_features, out_features, 1))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(self.activation_fn)
            in_features = out_features
        ## FC layers
        for j in range(n_layers):
            ## pick a higher dimension from the list
            pos = fc_dim_range.index(in_features)
            out_features = self.trial.suggest_categorical(f"out_features{j}", fc_dim_range[pos+1:])  
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(self.activation_fn)
            in_features = out_features
        
        self.layers.append(nn.Linear(in_features, self.k*self.k))

    def forward(self, xb):
        # initialize as identity
        bs = xb.shape[0]
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        # add identity to the output
        matrix = nn.Sequential(*self.layers)(xb).view(-1,self.k,self.k) + init
        return matrix

class Transform(nn.Module):
    def __init__(self, trial, activation_fn):
        super().__init__()
        self.trial = trial
        self.activation_fn = activation_fn
        self.k1 = self.trial.suggest_int("input_dim", low=2, high=128, log=True)
        self.k2 = self.trial.suggest_int("feat_dim", low=64, high=1024, log=True)
        self.input_transform = Tnet(self.trial, self.activation_fn, in_dim=3, k=self.k1)
        self.feature_transform = Tnet(self.trial, self.activation_fn, in_dim=self.k2, k=self.k2)
 
        self.mlp_layers1 = []
        self.mlp_layers2 = []
        n_layers = self.trial.suggest_int("n_layers", 2, 8, step=1)
        cnn_dim_range = [3, 16, 32, 64, 128, 256, 512, 1024, 2048]

        for i in range(n_layers):
            pos = cnn_dim_range.index(in_features)
            out_features = self.trial.suggest_categorical(f"out_features{j}", cnn_dim_range[pos+1:])  
            self.mlp_layers1.append(nn.conv1d(in_features, out_features,1))
            self.mlp_layers1.append(nn.BatchNorm1d(out_features))
            self.mlp_layers1.append(self.activation_fn)
            in_features = out_features
        
        for j in range(n_layers):
            pos = cnn_dim_range.index(in_features)
            out_features = self.trial.suggest_categorical(f"out_features{j}", cnn_dim_range[pos+1:])  
            self.mlp_layers2.append(nn.conv1d(in_features, out_features,1))
            self.mlp_layers2.append(nn.BatchNorm1d(out_features))
            self.mlp_layers2.append(self.activation_fn)
            in_features = out_features
        self.mlp_layers2.append(nn.MaxPool1d(in_features))
        self.mlp_layers2.append(nn.Flatten(1))

    def forward(self, xb):
        input_transform = self.input_transform(xb)
        feat_transform = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), input_transform).transpose(1,2)
        xb = nn.Sequential(*self.mlp_layers1)(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), feat_transform).transpose(1,2)
        output = nn.Sequential(*self.mlp_layers2)(xb)
        return output, input_transform, feat_transform

class PointNet(nn.Module):
    def __init__(self, trial, classes=222):
        super().__init__()
        self.trial = trial
        self.activation_fn = ACTIVATION[self.trial.suggest_categorical("relu", "elu", "gelu")]
        self.transform = Transform(self.trial, self.activation_fn)
        
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.LazyLinear(1024)
        self.fc3 = nn.LazyLinear(512)
        self.fc4 = nn.LazyLinear(256)
        self.fc5 = nn.LazyLinear(classes)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, xb):
        xb, input_transform, feat_transform = self.transform(input)     
        fc_layers = {'2048': self.fc1, '1024': self.fc2, '512': self.fc3, '256':self.fc4}
        bn_layers = {'2048': self.bn1, '1024': self.bn2, '512': self.bn3, '256':self.bn4}
        fc_dim_range = [2048, 1024, 512, 256]
        n_layers = self.trial.suggest_int("n_layers", 2, 4, step=1)
        in_features = xb.shape[-1]
        for i in n_layers:
            pos = fc_dim_range.index(in_features)
            out_features = self.trial.suggest_categorical(f"out_features{i}", fc_dim_range[pos+1:])
            xb = fc_layers[str(out_features)]
            xb = bn_layers[str(out_features)]
            xb = self.activation_fn(xb)
        outputs = self.fc5(xb)

        return outputs, input_transform, feat_transform


def objective(trial):
    train_set = PcDataset(train_file, num_points)
    val_set = PcDataset(val_file, num_points)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=4, drop_last=True)

    # Training Ops
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {DEVICE} . . .")

    net = PointNet(trial, classes=222)
    net.to(DEVICE)
    print(net)
  
    writer = SummaryWriter(log_dir=log_dir)

    model = PointNet(trial).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )
   
    trainer = Trainer(net, n_epochs, grad_accum_steps, optimizer, writer)
    val_loss = trainer.train(train_loader, val_loader)
    return val_loss


if __name__ == "__main__":
    n_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    num_points = int(sys.argv[3])
    train_file = sys.argv[4]
    val_file = sys.argv[5]
    log_dir = sys.argv[6]
    grad_accum_steps = int(sys.argv[7])

    study = optuna.create_study(directions="minimize")
    study.optimize(objective, n_trials=30)
