from numpy import block
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import torchvision.utils as vutils
import matplotlib.animation as animation

import web3
import json
import ipfshttpclient

import argparse
import ast
import random
import glob
import os
from tqdm import tqdm

from config import *
from utils import *
from models import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", help="Group creator", action="store_true")
    parser.add_argument("--group", help="ID of federated learning group", type=int)
    parser.add_argument("--dataroot", help="location of dataset", type=str)
    parser.add_argument("--rounds", help="number of training rounds", type=int, default=10)
    parser.add_argument("--epochs", help="epochs in each round", type=int, default=5)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0002)
    parser.add_argument("--device", help="gpu device id")
    parser.add_argument("--img-size", help="generate image size", type=int, default=64)
    parser.add_argument("--laten-dim", help="dimension of noise", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()
    print(args)
    group_id = args.group
    batch_size = args.batch_size
    laten_dim = args.laten_dim
    img_size = args.img_size
    rounds = args.rounds
    epochs = args.epochs
    pool_interval = 3
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)
    G_losses = []
    D_losses = []

    # initialize stage

    # setup IPFS client
    client = ipfshttpclient.connect(addr=IPFS_ADDR)

    # Configure web3
    web_3 = web3.Web3(web3.HTTPProvider(GETH_URL))
    print("Connected to block chain?", web_3.isConnected())
    web_3.eth.defaultAccount = web_3.eth.accounts[0] # authorize account to create transactions

    # connect with contract
    address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
    contract_ins = web_3.eth.contract(address=address, abi=json.loads(ABI)) # create contract instance

    # define event filters
    aggregate_filter = contract_ins.events.aggregation_complete.createFilter(fromBlock=0, toBlock='latest')
    model_filter = contract_ins.events.allModel_uploaded.createFilter(fromBlock=0, toBlock='latest')
    globalA_filter = contract_ins.events.global_accept.createFilter(fromBlock=0, toBlock='latest')
    globalR_filter = contract_ins.events.global_reject.createFilter(fromBlock=0, toBlock='latest')

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    print(f"Using device {device}")
    # load dataset
    dataset = dset.ImageFolder(root = args.dataroot,
                               transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.CenterCrop(img_size),
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
    print("==========================Grouping stage==========================")
    # create group or join group
    if args.creator: # create group
        contract_ins.functions.init_group(args.group).transact()
        id = 0
        print(f"Create group {group_id}, member id = {id}")
    
    else: # join group
        contract_ins.functions.join_group(args.group).transact()
        # time.sleep(5)
        # id = contract_ins.functions.get_memberID(group_id).call()
        id = 1
        print(f"Join group {group_id}, member id = {id}")

            
    # ============================================================
    # Training stage
    print("==========================Training stage==========================")

    # create checkpoint folder
    if not os.path.exists('runs'):
        os.mkdir('runs')


    iters = 0 
    img_list= []
    for round in range(1, rounds + 1):

        if not os.path.exists(os.path.join('runs', f'round{round}')):
            os.mkdir(os.path.join('runs', f'round{round}'))
        print('\n')
        print(f'                               training round {round}                               ')
        print('\n')
        best_g = 100
        best_d = 100

        for epoch in range(1, epochs + 1):

            gen_loss = 0
            dis_loss = 0

            for i, (img, _) in enumerate(tqdm(dataloader)):

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

                gen_loss += errG.item()
                dis_loss = dis_loss + errD_fake.item() + errD_real.item()

                # save the best result
                if errG.item() < best_g:
                    best_g = errG.item()
                    torch.save(generator.state_dict(), os.path.join('runs', f'round{round}', 'bestG.pt'))
                
                if (errD.item() < best_d):
                    best_d = errD.item()
                    torch.save(discriminator.state_dict(), os.path.join('runs', f'round{round}', 'bestD.pt'))

                G_losses.append(errG.item())
                D_losses.append(errD.item())
                if (iters % 500 == 0) or ((epoch == round) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = generator(noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # log training result
            print("epoch[{}/{}]:\nLoss_D: {:.4f} Loss_G: {:.4f}".format(epoch, epochs, dis_loss/len(dataloader), gen_loss/len(dataloader)))

        # upload model
        # save model state_dict to json format
        print("uploading model")
        gen = dict()
        dis = dict()

        for k, v in generator.state_dict().items():
            gen[k] = v.tolist()
        
        for k, v in discriminator.state_dict().items():
            dis[k] = v.tolist()
        
        gen_hs = client.add_json(gen)
        dis_hs = client.add_json(dis)

        print(f'Generator IPFS hash: {gen_hs}\nDiscriminator IPFS hash: {dis_hs}')

        contract_ins.functions.upload_model(gen_hs, dis_hs, round, group_id).transact()

        upload_finish = False

        print("Wait for all clients upload their models...")
        # wait for all clients upload their model
        while not upload_finish:
            for log in model_filter.get_new_entries():
                res = log['args']
                if res['round'] == round and res['group_id'] == group_id:
                    upload_finish = True
                time.sleep(pool_interval)
        
        # ===========================================================================
        # Aggregate stage
        print("==========================Aggregation stage==========================")
        aggreator_id = contract_ins.functions.get_aggregater(group_id).call()
        total = contract_ins.functions.get_member_count(group_id).call()
        count = int(total / 2)
        aggregate = False

        # this client is the chosen aggreator
        if aggreator_id == id:
            print("Do aggregation")
            G_hs, D_hs = contract_ins.functions.fetch_model(round, group_id).call()

            # fetch files from IPFS and do Federated Average
            global_G = FedAvg(client, G_hs)
            global_D = FedAvg(client, D_hs)

            # upload global model to IPFS
            GG_hs = client.add_json(global_G)
            GD_hs = client.add_json(global_D)

            print(f'Global generator hash :{GG_hs}\nGlobal Discriminator hash: {GD_hs}')
            contract_ins.functions.upload_globalModel(GG_hs, GD_hs, round, group_id).transact()

            # generate random id for validation
            val_id = random.sample(range(total), count)
            contract_ins.functions.choose_validator(group_id, round, val_id).transact()


        # wait for aggregation complete
        print("Wait for aggregation complete...")
        aggregation_complete = False

        # wait for aggregation complete
        while not aggregation_complete:
            for log in aggregate_filter.get_new_entries():
                res = log['args']
                if res['round'] == round and res['group_id'] == group_id: # aggregation of this round has completed
                    aggregation_complete = True
            time.sleep(pool_interval)

        # ========================================================
        # validate stage
        print("==========================Validation stage==========================")
        validators = contract_ins.functions.get_validator(group_id, round).call()
        valid = False
        # choose to be validator this round
        if id in validators:
            print("Run validation process")
            valid = True
            global_gen, global_dis = contract_ins.functions.fetch_global_model(round, group_id).call()
            global_gen = load_model(client, global_gen)
            global_dis = load_model(client, global_dis)
            
            generator.load_state_dict(global_gen)
            discriminator.load_state_dict(global_dis)

            generator.to(device)
            discriminator.to(device)

            # run through training set once
            generator.eval()
            discriminator.eval()
            global_errG = 100
            global_errD = 100

            for i, (img, _) in enumerate(tqdm(dataloader)):
                with torch.no_grad():
                    # calculate discriminator loss
                    b_size = img.size(0)
                    real_cpu = img.to(device)
                    label = torch.full((b_size,), 1., dtype=torch.float, device=device)
                    output = discriminator(real_cpu).view(-1)
                    errD = criterion(output, label)
                    errD_real = errD.item()

                    noise = torch.randn(b_size, laten_dim, 1, 1, device=device)
                    fake = generator(noise)
                    label.fill_(0.)
                    output = discriminator(fake).view(-1)
                    errD = criterion(output, label)
                    errD_fake = errD.item()
                    errD = errD_fake + errD_real

                    # calculate generator loss
                    label.fill_(1.)
                    output = discriminator(fake).view(-1)
                    errG = criterion(output, label)
                    errG = errG.item()

                    global_errG = errG if errG < global_errG else global_errG
                    global_errD = errD if errD < global_errD else global_errD

            # calculate performance of global model
            if abs(global_errD - best_d) <= 1 and abs(global_errG - best_g) <= 1:
                accept = 1

            else:
                accept = 0

            # vote according to validation result
            contract_ins.functions.vote(accept, group_id, round, count).transact()
            
        
        # wait for validation complete
        print("Wait for validation...")
        validation_complete = False
        global_accept = False
        while not validation_complete:
            for log in globalA_filter.get_new_entries():
                res = log['args']
                if res['round'] == round:
                    global_accept = True
                    validation_complete = True
                
            for log in globalR_filter.get_new_entries():
                res = log['args']
                if res['round'] == round:
                    validation_complete = True
            time.sleep(pool_interval)
                
        # reload model from best checkpoint
        if not global_accept:
            # load global model weight
            print("reject global model, continue training with local weight")
            gen_weight = torch.load(os.path.join('runs', f'round{round}', 'bestG.pt'), map_location=device)
            dis_weight = torch.load(os.path.join('runs', f'round{round}', 'bestD.pt'), map_location=device)
            generator.load_state_dict(gen_weight)
            discriminator.load_state_dict(dis_weight)

        # load global model
        if global_accept and not valid:
            print("global accept, continue training with global model")
            global_gen, global_dis = contract_ins.functions.fetch_global_model(round, group_id).call()
            generator.load_state_dict(load_model(client, global_gen))
            discriminator.load_state_dict(load_model(client, global_dis))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()





