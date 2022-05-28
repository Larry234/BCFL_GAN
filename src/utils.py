import web3
import time

def handle_event(event):

    return web3.Web3.toJSON(event)


def log_loop(event_filter, pool_iterval=10):    

    while True:
        for PairCreated in event_filter.get_new_entries():
            handle_event(PairCreated)
        time.sleep(pool_iterval)
        


def upload_model(weight_file, client):

    res = client.add(weight_file)
    return res

def create_group(contract, max_registry, group_id):

    contract.function.init_group(max_registry, group_id).transact()

def join_group(contract, group_id):

    contract.function.join_group(group_id).transact()


