import torch
import torch.nn as nn

import web3
import json
import ipfshttpclient

import argparse

from config import *




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", help="Group creator", action="store_true")
    parser.add_argument("--group", "ID of federated learning group", type=int)

    args = parser.parse_args()

    # Configure web3
    web_3 = web3.Web3(web3.HTTPProvider(GETH_URL))

    # connect with contract
    address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
    contract = web_3.eth.contract(address=address, abi=ABI)

    # create group or join group
    if args.creator:
        


