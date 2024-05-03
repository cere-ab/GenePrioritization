from pathlib import Path
from utils.dataclass import getData
from model.model import GNN7L_SAGEConv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import json


class Trainer:
    """
    <Purpose>

    Attributes
    ----------
    atr1 : dtype
        <purpose>

    Methods
    -------
    __init__():
        Constructor
    m(p):
        <purpose>
    
    """
    device = None

    def __init__(self, configFile):
        super(Trainer, self).__init__()

        with open(configFile, "r") as cfg:
            self.config = json.load(cfg)
        Trainer.device = self.config["hw"] if torch.cuda.is_available() else 'cpu'

        self.epochs = self.config['epochs']
        
        #NOTE: Both the loss and the optimizer should ideally be separate classes placed under the model directory!
        self.optimizer = None
        self.lossfn = None

    def setOptimAndLossFn(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        self.lossfn = F.nll_loss

    def train(self, model, data, labels, train_mask):
        model.train()
        self.optimizer.zero_grad()
        
        logits = model(data)
        output = logits.argmax(1)

        train_loss = self.lossfn(logits[train_mask], labels[train_mask])
        
        train_acc = (output[train_mask] == labels[train_mask]).float().mean()

        train_loss.backward()
        self.optimizer.step()

        return (train_acc, train_loss)

    def validate(self, model, data, labels, val_mask):
        model.eval()
        
        logits = model(data)
        output = logits.argmax(1)

        val_loss = self.lossfn(logits[val_mask], labels[val_mask])

        val_acc = (output[val_mask] == labels[val_mask]).float().mean()

        return (val_acc, val_loss)

    def save_model(self, model, e):
        torch.save(
            model.state_dict(),
            self.config['projRootPath']+
            self.config['modelsPath']+
            "GNN7L_SAGEConv_"+
            str(e)+
            ".pt"
        )


def main():
    """
    <Purpose>

    Parameter
    ---------
    arg : dtype
        This function does not take any parameter.

    Raises
    ------
    err
        This function does not raise any error/exception.

    Returns
    -------
    dtype
        This function does not return any value.

    """
    cfgFile = str(Path(__file__).parent.absolute() / "configFiles/train_config.json")
    DLobj = Trainer(cfgFile)    #NOTE: Have to advance define the Trainer object to be able to utilize the __init__() config!

    # Preparing the data object
    print("Starting the file {}\n".format(str(__file__)))
    if Trainer.device != 'cpu':
        print("using GPU...")
    print('\npreparing the data...', end='')
    dataGraph_filename = "C0006142_nedbit.gml"
    data, _ = getData(cfgFile, dataGraph_filename)
    data = data.to(Trainer.device)
    print('done !')

    # Creating the model object
    print('preparing the model...', end='')
    model = GNN7L_SAGEConv(
        in_features=data.num_features,
        out_classes=data.num_classes
    ).to(Trainer.device)
    print('done !')

    # Creating an object for the trainer class
    print('preparing the trainer...', end='')
    
    DLobj.setOptimAndLossFn(model)  #FIXME: Bad design!

    train_mask = data['train_mask']
    val_mask = data['val_mask']

    labels = data.y
    output = None

    # train_acc_curve = []
    # train_lss_curve = []

    best_train_acc = torch.tensor([0]).to(Trainer.device)
    best_val_acc = torch.tensor([0]).to(Trainer.device)
    best_train_lss = torch.tensor([999]).to(Trainer.device)
    best_loss_epoch = torch.tensor([0]).to(Trainer.device)
    print('done !')

    # Main loop for iterations (Training and Validation)
    #NOTE: Tesing will be done in separate file
    print('Begin train and validate cycles for {} epochs...'.format(DLobj.epochs))
    pbar = tqdm(total=DLobj.epochs+1)
    for e in range(DLobj.epochs+1):
        train_acc, train_loss = DLobj.train(model, data, labels, train_mask)
        
        # train_acc_curve.append(train_acc.item())
        # train_lss_curve.append(train_loss.item())

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        
        val_acc, val_loss = DLobj.validate(model, data, labels, val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if train_loss < best_train_lss:
            best_train_lss = train_loss
            best_loss_epoch = torch.tensor([e], dtype=torch.int32).to(Trainer.device)
            DLobj.save_model(model, e)

        if e % 5000 == 0 or e == DLobj.epochs:
            # print('[Epoch: {:04d}]'.format(e),
            # 'train loss: {:.4f},'.format(train_loss.item()),
            # 'train acc: {:.4f},'.format(train_acc.item()),
            # 'val loss: {:.4f},'.format(val_loss.item()),
            # 'val acc: {:.4f} '.format(val_acc.item()),
            # '(best train acc: {:.4f},'.format(best_train_acc.item()),
            # 'best val acc: {:.4f},'.format(best_val_acc.item()),
            # 'best train loss: {:.4f} '.format(best_train_lss.item()),
            # '@ epoch', best_loss_epoch ,')')
            pbar.write(
                '[Epoch: {:04d}] train loss: {:.4f} train acc: {:.4f} | val loss: {:.4f} val acc: {:.4f} | (best train acc: {:.4f} best val acc: {:.4f} best train loss: {:.4f} @ epoch {:04d}'.format(
                    e, train_loss.item(), train_acc.item(), 
                    val_loss.item(), val_acc.item(), 
                    best_train_acc.item(), best_val_acc.item(), 
                    best_train_lss.item(), best_loss_epoch.item()
                )
            )
        pbar.update(1)  # update the progress bar, per iteration


if __name__ == '__main__':
    main()
