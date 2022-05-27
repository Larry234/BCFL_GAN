import json
from web3 import Web3
import asyncio

geth_url = ""
web3 = Web3(Web3.HTTPProvider(geth_url))

abi = json.loads('[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},
                   {"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"aggregater","type":"address"}],"name":"aggregater_selected","type":"event"},
                   {"anonymous":false,"inputs":[],"name":"allModel_uploaded","type":"event"},
                   {"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"creator","type":"address"},{"indexed":false,"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"createGroup","type":"event"},
                   {"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"global_hs","type":"string"}],"name":"fetch_global","type":"event"},
                   {"anonymous":false,"inputs":[],"name":"stopTraining","type":"event"},
                   {"inputs":[{"internalType":"uint256","name":"round","type":"uint256"}],"name":"fetch_global_model","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"round","type":"uint256"},{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"fetch_model","outputs":[{"internalType":"string[]","name":"","type":"string[]"}],"stateMutability":"view","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"get_aggregater","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"MaxRegistry","type":"uint256"},{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"init_group","outputs":[],"stateMutability":"nonpayable","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"uint256","name":"","type":"uint256"}],"name":"items","outputs":[{"internalType":"address","name":"sender","type":"address"},{"internalType":"string","name":"IPFS_hash","type":"string"}],"stateMutability":"view","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"join_group","outputs":[],"stateMutability":"nonpayable","type":"function"},
                   {"inputs":[{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"leave_group","outputs":[],"stateMutability":"nonpayable","type":"function"},
                   {"inputs":[{"internalType":"string","name":"hs","type":"string"},{"internalType":"uint256","name":"round","type":"uint256"},{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"upload_global_model","outputs":[],"stateMutability":"nonpayable","type":"function"},
                   {"inputs":[{"internalType":"string","name":"hs","type":"string"},{"internalType":"uint256","name":"round","type":"uint256"},{"internalType":"uint256","name":"group_id","type":"uint256"}],"name":"upload_model","outputs":[],"stateMutability":"nonpayable","type":"function"}]')

address = web3.toChecksumAddress("")

contract = web3.eth.contract(address=address, abi=abi)

# define function to handle events and print to the console
def handle_event(event):
    print(Web3.toJSON(event))
    # and whatever
    
    # asynchronous defined function to loop
# this loop sets up an event filter and is looking for new entires for the "PairCreated" event
# this loop runs on a poll interval
async def log_loop(event_filter, poll_interval):
    while True:
        for PairCreated in event_filter.get_new_entries():
            handle_event(PairCreated)
        await asyncio.sleep(poll_interval)


# when main is called
# create a filter for the latest block and look for the "PairCreated" event for the uniswap factory contract
# run an async loop
# try to run the log_loop function above every 2 seconds
def main():
    event_filter = contract.events.PairCreated.createFilter(fromBlock='latest')
    #block_filter = web3.eth.filter('latest')
    # tx_filter = web3.eth.filter('pending')
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            asyncio.gather(
                log_loop(event_filter, 2)))
                # log_loop(block_filter, 2),
                # log_loop(tx_filter, 2)))
    finally:
        # close loop to free up system resources
        loop.close()


if __name__ == "__main__":
    main()