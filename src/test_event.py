from web3 import Web3
import json
from config import *
from utils import *

web_3 = Web3(Web3.HTTPProvider(GETH_URL, request_kwargs={'verify': False}))
# print(web_3.isConnected())
# connect with contract
address = web_3.toChecksumAddress(CONTRACT_ADDRESS)
contract_ins = web_3.eth.contract(address=address, abi=json.loads(ABI)) # create contract instance
web_3.eth.defaultAccount = web_3.eth.accounts[0]
a = contract_ins.functions.get_aggregater(0).call()
print(a)
# training_filter = contract_ins.events.startTraining.createFilter(fromBlock=0, toBlock='latest')
# while True:
#         for event_log in training_filter.get_all_entries():
#             # handle event log
#             res = web3.Web3.toJSON(event_log)
#             print(res)