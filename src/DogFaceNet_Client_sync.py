import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import web3
import json
import ipfshttpclient

import argparse

from config import *
from utils import *
from models import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", help="Group creator", action="store_true")
    parser.add_argument("--group", help="ID of federated learning group", type=int)
    parser.add_argument("--registry", help="number of registries", type=int)
    parser.add_argument("--rounds", help="number of training rounds", type=int, default=10)
    parser.add_argument("--epochs", help="epochs in each round", type=int, default=5)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--device", help="gpu device id")
    parser.add_argument("--img-size", help="generate image size", type=int, default=64)
    parser.add_argument("--laten-dim", help="dimension of noise", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()
    batch_size = args.batch_size
    laten_dim = args.laten_dim
    img_size = args.img_size
    rounds = args.rounds
    epochs = args.epochs

    # initialize stage
    # Configure web3
    web_3 = web3.Web3(web3.HTTPProvider(GETH_URL))

    # connect with contract
    address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
    contract_ins = web_3.eth.contract(address=address, abi=ABI) # create contract instance

    # define event filters
    training_filter = contract_ins.events.StartTraining.createFilter(fromBlock='latest')
    aggregate_filter = contract_ins.events.aggregater_selected.createFilter(fromBlock='latest')
    model_filter = contract_ins.events.allModel_uploaded.createFilter(fromBlock='latest')

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    # load dataset
    transform =transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = torchvision.datasets.ImageFolder(DATAROOT, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds)

    # define model
    generator = Generator(laten_dim=laten_dim)
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    # define criterion
    criterion = nn.BCELoss()
    criterion.to(device)

    # setup optimizer
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ==========================================================
    # Grouping stage

    # create group or join group
    if args.creator: # create group
        create_group(contract_ins, args.regisrty, args.group)
    
    else: # join group
        join_group(contract_ins, args.group)

    while True:
        for event_log in training_filter.get_new_entries():
            # handle event log
            res = web3.Web3.toJSON(event_log)
            
    

    # ============================================================
    # Training stage
    for round in range(rounds):

        for epoch in range(epochs):
            noise = torch.rand(batch_size, laten_dim, 1, 1)





    
    # Aggregate stage







