// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(uint => Item[]) public gen_models;
    mapping(uint => Item[]) public dis_models;
    mapping(uint => Group) groups;
    mapping(uint => uint[]) votes;
    mapping(uint => uint[]) validators;

    uint[][] modelG_count;
    uint[][] modelD_count;
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
    event allModel_uploaded(uint round);
    event aggregation_complete(uint round, uint group_id);
    event global_accept(uint round, uint group);
    event global_reject(uint round, uint group);


    function init_group(uint MaxRegistry, uint group_id) public {

        address[] memory members = new address[](MaxRegistry);
        members[0] = msg.sender;
        Group memory initial_group = Group(1, 0, members);
        groups[group_id] = initial_group;
    }

    function join_group(uint group_id) public {

        groups[group_id].member_count += 1;
        groups[group_id].members.push(msg.sender); // add new member into address array

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

    function get_memberID(uint group_id) public view returns(uint){

        return groups[group_id].member_count - 1;
    }

    function upload_genModel(string memory hs, uint round, uint group_id) public {

        require(modelG_count[group_id][round] < groups[group_id].member_count, "too discriminator models uploaded!");
        modelG_count[group_id][round]++;
        Item memory item = Item(msg.sender, hs);
        gen_models[round].push(item);

        // check if all clients upload their model
        if (gen_models[round].length == groups[group_id].member_count && dis_models[round].length == groups[group_id].member_count) 
        {
            emit allModel_uploaded(round);
        }

    }

    function upload_disModel(string memory hs, uint round, uint group_id) public {

        require(modelD_count[group_id][round] < groups[group_id].member_count, "too discriminator models uploaded!");
        modelD_count[group_id][round]++;
        Item memory item = Item(msg.sender, hs);
        dis_models[round].push(item);

        // check if all clients upload their model
        if (gen_models[round].length == groups[group_id].member_count && dis_models[round].length == groups[group_id].member_count) 
        {
            emit allModel_uploaded(round);
        }

    }

    function upload_global_genModel(string memory hs, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        string memory empty = "";

        // only chosen aggregater can upload global model
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!"); 

        // global model cannot be uploaded more than once
        require(keccak256(abi.encodePacked(global_genModels[round])) == keccak256(abi.encodePacked(empty)));
        global_genModels[round] = hs; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        if (keccak256(abi.encodePacked(global_disModels[round])) != keccak256(abi.encodePacked(empty)))
        {
            emit aggregation_complete(round, group_id);
        }
        
    }

    function upload_global_disModel(string memory hs, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        string memory empty = "";

        // only chosen aggregater can upload global model
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!");

        // global model cannot be uploaded more than once
        require(keccak256(abi.encodePacked(global_disModels[round])) == keccak256(abi.encodePacked(empty)));

        global_disModels[round] = hs; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        if (keccak256(abi.encodePacked(global_genModels[round])) != keccak256(abi.encodePacked(empty)))
        {
            emit aggregation_complete(round, group_id);
        }
        
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

    function vote(uint res, uint group_id, uint round, uint voters) public {

        votes[round].push(res);

        uint accepts = 0;

        // every validators have participated in the voting of this round
        if (votes[round].length == voters) {

            for (uint i = 0; i < votes[round].length; i++) {
                if (votes[round][i] == 1) accepts++; // client accept this global model
            }
            
            // Global model is accepted
            if (accepts >= (voters / 2)) {
                emit global_accept(round, group_id);
            }

            // Global model is rejected
            else {
                emit global_reject(round, group_id);
            }
        }
    }

    function choose_validator(uint round, uint[] memory vals) public {
        
        for (uint i = 0; i < vals.length; i++) {
            validators[round].push(vals[i]);
        }
    }

    function get_validator(uint round) public view returns (uint[] memory) {

        return validators[round];
    }


    // ============================================================================
    // utility functions for debugging

    function get_genCount(uint round) public view returns (uint) {
        return gen_models[round].length;
    }

    function get_disCount(uint round) public view returns (uint) {
        return dis_models[round].length;
    }

    function get_member_count(uint group_id) public view returns (uint) {
        return groups[group_id].member_count;
    }

}