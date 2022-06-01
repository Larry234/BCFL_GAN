import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as dset

import web3
import json
import ipfshttpclient

import argparse
import random
import glob
from tqdm import tqdm

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
    aggregate_filter = contract_ins.events.aggregater_selected.createFilter(fromBlock=0, toBlock='latest')
    model_filter = contract_ins.events.allModel_uploaded.createFilter(fromBlock=0, toBlock='latest')
    globalA_filter = contract_ins.events.global_accept.createFilter(fromBlock=0, toBlock='latest')
    globalR_filter = contract_ins.events.global_reject.createFilter(fromBlock=0, toBlock='latest')

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
    for round in range(1, rounds + 1):
        print(f'============================== round {round} ==============================')
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
                dis_loss += errD.item()

                # save the best result
                if errG.item() < best_g:
                    best_g = errG.item()
                
                if (errD.item() < best_d):
                    best_d = errD.item()

            # log training result
            print("epoch[{}/{}]:\nLoss_D: {:.4f} Loss_G: {:.4f}".format(epoch, epochs, dis_loss/len(dataloader), gen_loss/len(dataloader)))

        # upload model
        # save model state_dict to json format
        gen = dict()
        dis = dict()

        for k, v in generator.state_dict().items():
            gen[k] = v.tolist()
        
        for k, v in discriminator.state_dict().items():
            dis[k] = v.tolist()
        
        gen_hs = client.add_json(gen)
        dis_hs = client.add_json(dis)

        contract_ins.functions.upload_genModel(gen_hs, round, group_id).transact()
        contract_ins.functions.upload_disModel(dis_hs, round, group_id).transact()

        upload_finish = False    

        print("Wait for all clients upload their models...")
        # wait for all clients upload their model
        while not upload_finish:
            for log in model_filter.get_new_entries():
                res = log[0]['args']
                if res['round'] == round:
                    upload_finish = True
                time.sleep(pool_interval)
        
        # =========================================================
        # Aggregate stage
        aggreator_id = contract_ins.functions.get_aggreator(group_id).call()
        total = contract_ins.functions.get_member_count(group_id).call()
        count = int(total / 2)
        model_list = list()

        # this client is the chosen aggreator
        if aggreator_id == id: 
            models_hs = contract_ins.functions.fetch_model(round, group_id).call()

            # fetch files from IPFS
            for hs in models_hs:
                client.cat(hs)

            # take average of model weights

            # generate random id for validation
            val_id = random.sample(range(total), count)
            contract_ins.functions.choose_validator(round, val_id).transact()


        # wait for aggregation complete
        else:   
            aggregation_complete = False

            # wait for aggregation complete
            print("Wait for aggregation complete...")
            while not aggregation_complete:
                for log in aggregate_filter.get_new_entries():
                    res = log[0]['args']
                    if res['round'] == round: # aggregation of this round has completed
                        aggregation_complete = True

        # ========================================================
        # validate stage

        validators = contract_ins.functions.get_validator(round).call()

        # choose to be validator this round
        if id in validators:
            global_gen, global_dis = contract_ins.functions.fetch_global_model(round).call()
            global_gen = client.get_json(global_gen)
            global_dis = client.get_json(global_dis)

            # load global model weight
            gen_weight = generator.state_dict()
            dis_weight = discriminator.state_dict()

            for k, v in gen_weight.items():
                gen_weight[k] = torch.FloatTensor(global_gen[k])
            
            for k, v in dis_weight.items():
                dis_weight[k] = torch.FloatTensor(global_dis[k])
            
            generator.load_state_dict(gen_weight)
            discriminator.load_state_dict(dis_weight)

            generator.to(device)
            discriminator.to(device)

            # run through training set once
            print("Run validation process")
            generator.eval()
            discriminator.eval()
            for i, (img, _) in enumerate(tqdm(dataloader)):
                with torch.no_grad():
                    # calculate discriminator loss
                    b_size = img.size(0)
                    real_cpu = img.to(device)
                    label = torch.full((b_size,), 1., dtype=torch.float, device=device)
                    output = discriminator(real_cpu).view(-1)
                    errD = criterion(output, label)
                    errD_real = output.mean().item()

                    noise = torch.randn(b_size, laten_dim, 1, 1, device=device)
                    fake = generator(noise)
                    label.fill_(0.)
                    output = discriminator(fake).view(-1)
                    errD = criterion(output, label)
                    errD_fake = errD.mean().item()
                    errD = (errD_fake + errD_real) / 2

                    # calculate generator loss
                    label.fill(1.)
                    output = discriminator(fake).view(-1)
                    errG = criterion(output, label)
                    errG = errG.mean().item()

            
            accept = 1

            # vote
            contract_ins.functions.vote(accept, group_id, round, count).transact()
            
        
        # wait for validation complete
        validation_complete = False
        global_accept = False
        while not validation_complete:
            for log in globalA_filter.get_new_entries():
                res = log[0]['args']
                if res['round'] == round:
                    global_accept = True
                    validation_complete = True
                
            for log in globalR_filter.get_new_entries():
                res = log[0]['args']
                if res['round'] == round:
                    validation_complete = True
                







