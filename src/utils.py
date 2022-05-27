def upload_model(weight_file, client):

    res = client.add(weight_file)
    return res

def create_group(contract, max_registry, group_id):

    contract.function.init_group(max_registry, group_id).transact()

def join_group(contract, group_id):

    contract.function.join_group(group_id).transact()