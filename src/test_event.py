from web3 import Web3
import json
from config import *
from utils import *

w3 = Web3(Web3.HTTPProvider(GETH_URL))
# print(web_3.isConnected())
# connect with contract
address = w3.toChecksumAddress(CONTRACT_ADDRESS)
contract_ins = w3.eth.contract(address=address, abi=json.loads(ABI)) # create contract instance
w3.eth.defaultAccount = w3.eth.accounts[0]
a = contract_ins.functions.get_aggregater(0).call()
print(a)
# training_filter = contract_ins.events.startTraining.createFilter(fromBlock=0, toBlock='latest')
# while True:
#         for event_log in training_filter.get_all_entries():
#             # handle event log
#             res = web3.Web3.toJSON(event_log)
#             print(res)