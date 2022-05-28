import torch
import torch.nn as nn

import web3
import json
import ipfshttpclient

import argparse

from config import *
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", help="Group creator", action="store_true")
    parser.add_argument("--group", "ID of federated learning group", type=int)
    parser.add_argument("--registry", "number of registries", type=int)
    parser.add_argument("--round", "number of training rounds", type=int)

    args = parser.parse_args()

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





    
    # Aggregate stage







