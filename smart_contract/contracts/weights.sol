// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(uint => Item[]) public gen_models;
    mapping(uint => Item[]) public dis_models;
    mapping(uint => Group) groups;
    mapping(uint => uint[]) votes;

    int group_counts;

    uint[] group_ids;
    string[] global_genModels;
    string[] global_disModels;

    // data structures
    struct Item {
        address sender;
        string IPFS_hash;
    }

    struct Group {
        uint member_count;
        uint aggregater_id;
        address[] members;
    }
    
    // events
    event startTraining(uint group_id);
    event allModel_uploaded();
    event aggregation_complete(uint round, uint group_id, address aggregater);
    event global_accept();
    event fetch_global(uint group_id, uint round);
    event stopTraining();

    constructor() {
        group_counts = 0;
    }

    function init_group(uint MaxRegistry, uint group_id) public returns(uint) {

        address[] memory members = new address[](MaxRegistry);
        members[0] = msg.sender;
        Group memory initial_group = Group(1, 0, members);
        groups[group_id] = initial_group;
        group_ids.push(group_id);
        group_counts++;
        return groups[group_id].member_count - 1; // return his id in the group to client 
    }

    function join_group(uint group_id) public returns(uint){

        groups[group_id].member_count += 1;
        groups[group_id].members.push(msg.sender); // add new member into address array

        if (groups[group_id].member_count == groups[group_id].members.length) {
            emit startTraining(group_id);
        }

        return groups[group_id].member_count - 1;
    }

    function leave_group(uint group_id) public {
        
        Group memory group = groups[group_id];
        uint del_index;

        // find member address and delete from the group
        for (uint i = 0; i < group.member_count; i++){
            if (group.members[i] == msg.sender){
                del_index = i;
                delete groups[group_id].members[i];
                break;
            }
        }

        // fill the gap generate from delete element
        for (uint i = del_index; i < group.member_count - 1; i++){
            groups[group_id].members[i] = groups[group_id].members[i+1]; // shift array
        }

        groups[group_id].member_count --;
    }

    function upload_genModel(string memory hs, uint round, uint group_id) public {

        require(gen_models[round].length < groups[group_id].member_count, "too discriminator models uploaded!");
        Item memory item = Item(msg.sender, hs);
        gen_models[round].push(item);

        // check if all clients upload their model
        if (gen_models[round].length == groups[group_id].member_count && dis_models[round].length == groups[group_id].member_count) 
        {
            emit allModel_uploaded();
        }

    }

    function upload_disModel(string memory hs, uint round, uint group_id) public {

        require(dis_models[round].length < groups[group_id].member_count, "too discriminator models uploaded!");
        Item memory item = Item(msg.sender, hs);
        dis_models[round].push(item);

        // check if all clients upload their model
        if (gen_models[round].length == groups[group_id].member_count && dis_models[round].length == groups[group_id].member_count) 
        {
            emit allModel_uploaded();
        }

    }

    function upload_global_genModel(string memory hs, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!");

        global_genModels[round] = hs; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        // trigger event to let client update model
        // emit fetch_global(hs);
        
    }

        function upload_global_disModel(string memory hs, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!");

        global_disModels[round] = hs; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        // trigger event to let client update model
        // emit fetch_global(hs);
        
    }


    function fetch_model(uint round, uint group_id) public view returns(string [] memory m1, string [] memory m2) {

        uint data_count = groups[group_id].member_count;
        string[] memory gen_ret = new string[](data_count);
        string[] memory dis_ret = new string[](data_count);

        // fetch all model stored in contract
        for (uint i = 0; i < data_count; i++) {
            gen_ret[i] = gen_models[round][i].IPFS_hash;
            dis_ret[i] = dis_models[round][i].IPFS_hash;
        }
        return (gen_ret, dis_ret);
    }

    function fetch_global_model(uint round) public view returns (string memory gen, string memory dis) {

        return (global_genModels[round], global_disModels[round]);
    }


    function get_aggregater(uint group_id) public view returns(uint) {

        return groups[group_id].aggregater_id;
    }

    function get_max_member(uint group_id) public view returns(uint) {
        return groups[group_id].members.length;
    }

    function get_member_count(uint group_id) public view returns(uint) {
        return groups[group_id].member_count;
    }


    function vote(uint res, uint group_id, uint round) public {

        votes[round].push(res);

        uint accepts = 0;
        if (votes[round].length == groups[group_id].member_count) {

            for (uint i = 0; i < votes[round].length; i++) {
                if (votes[round][i] == 1) accepts++; // client accept this global model
            }

            if (accepts > groups[group_id].member_count / 2) {
                emit global_accept();
            }
        }
    }

    // function validate() public {

    // }

}