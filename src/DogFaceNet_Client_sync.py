import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as dset

import web3
import json
import ipfshttpclient

import argparse
import glob

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
    group_id = args.group
    batch_size = args.batch_size
    laten_dim = args.laten_dim
    img_size = args.img_size
    rounds = args.rounds
    epochs = args.epochs
    pool_interval = 3

    # initialize stage

    # setup IPFS client
    client = ipfshttpclient.connect(addr=IPFS_ADDR)

    # Configure web3
    web_3 = web3.Web3(web3.HTTPProvider(GETH_URL))
    web_3.eth.defaultAccount = web_3.eth.accounts[0]

    # connect with contract
    address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
    contract_ins = web_3.eth.contract(address=address, abi=ABI) # create contract instance

    # define event filters
    training_filter = contract_ins.events.startTraining.createFilter(fromBlock='latest')
    aggregate_filter = contract_ins.events.aggregater_selected.createFilter(fromBlock='latest')
    model_filter = contract_ins.events.allModel_uploaded.createFilter(fromBlock='latest')

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    # load dataset
    image_size = 64
    dataset = dset.ImageFolder(root = DATAROOT,
                                transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    # define model
    generator = Generator(laten_dim=laten_dim)
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    # apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)

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
        id = create_group(contract_ins, args.regisrty, args.group)
    
    else: # join group
        id = join_group(contract_ins, args.group)

            
    # ============================================================
    # Training stage
    for round in range(rounds):

        for epoch in range(epochs):

            for i, (img, _) in enumerate(dataloader):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                discriminator.zero_grad()
                # Format batch
                real_cpu = img.to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), 1., dtype=torch.float, device=device)

                # Forward pass real batch through D
                output = discriminator(real_cpu).view(-1)

                # Calculate loss on all-real batch
                errD_real = criterion(output, label)

                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, laten_dim, 1, 1, device=device)

                # Generate fake image batch with G
                fake = generator(noise)
                label.fill_(0.)
                # Classify all fake batch with D
                output = discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                label.fill_(1.)  # fake labels are real for generator cost

                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(fake).view(-1)

                # Calculate G's loss based on this output
                errG = criterion(output, label)

                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()

                # Update G
                optimizerG.step()
        
        # upload model

        # Aggregate stage
        aggreator_id = contract_ins.functions.get_aggreator(group_id).call()

        model_list = list()

        # this client is the chosen aggreator
        if aggreator_id == id: 

            models_hs = contract_ins.functions.fetch_model(round, group_id).call()

            # fetch files from IPFS
            for hs in models_hs:
                client.get(hs)

            # load weight files

        # wait for aggregation complete
        else:
            
            while True:
                for log_pair in aggregate_filter.get_new_entries():
                    log_dict = log_pair
                time.sleep(pool_interval)
                break

        # validate stage







