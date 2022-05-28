import web3
import json
from config import *
from utils import *

web_3 = web3.Web3(web3.HTTPProvider(GETH_URL))
# connect with contract
address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
contract_ins = web_3.eth.contract(address=address, abi=ABI) # create contract instance

training_filter = contract_ins.events.startTraining.createFilter(fromBlock='latest')
while True:
        for event_log in training_filter.get_new_entries():
            # handle event log
            res = web3.Web3.toJSON(event_log)
            print(res)