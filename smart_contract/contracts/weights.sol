// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(uint => Item[]) public gen_models;
    mapping(uint => Item[]) public dis_models;
    mapping(uint => Group) groups;
    mapping(uint => uint[]) validators;

    uint[50][8] modelG_count;
    uint[50][8] modelD_count;
    string[50][8] global_genModels;
    string[50][8] global_disModels;
    uint[50][8] vote_result;
    uint[50][8] vote_count;

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
    event allModel_uploaded(uint round, uint group_id);
    event aggregation_complete(uint round, uint group_id);
    event global_accept(uint round, uint group);
    event global_reject(uint round, uint group);

    function init_group(uint group_id) public {

        Group memory initial_group;
        initial_group.aggregater_id = 0;
        initial_group.member_count = 1;
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

    function upload_model(string memory gen, string memory dis, uint round, uint group_id) public {

        require(modelG_count[group_id][round] < groups[group_id].member_count, "too generator models uploaded!");
        modelG_count[group_id][round] += 1;
        Item memory item = Item(msg.sender, gen);
        gen_models[round].push(item);

        require(modelD_count[group_id][round] < groups[group_id].member_count, "too discriminator models uploaded!");
        modelD_count[group_id][round] += 1;
        Item memory item1 = Item(msg.sender, dis);
        dis_models[round].push(item1);

        // check if all clients upload their model
        if (modelD_count[group_id][round] == groups[group_id].member_count && modelG_count[group_id][round] == groups[group_id].member_count) 
        {
            emit allModel_uploaded(round, group_id);
        }

    }

    function upload_globalModel(string memory gen, string memory dis, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        string memory empty = "";

        // only chosen aggregater can upload global model
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!"); 

        // global model cannot be uploaded more than once
        require(keccak256(abi.encodePacked(global_genModels[group_id][round])) == keccak256(abi.encodePacked(empty)), "Cannot upload global model more than once!");
        global_genModels[group_id][round] = gen; // upload global model

        // global model cannot be uploaded more than once
        require(keccak256(abi.encodePacked(global_disModels[group_id][round])) == keccak256(abi.encodePacked(empty)), "Cannot upload global model more than once!");
        global_disModels[group_id][round] = dis; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        emit aggregation_complete(round, group_id);
        
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

    function fetch_global_model(uint round, uint group_id) public view returns (string memory gen, string memory dis) {

        return (global_genModels[group_id][round], global_disModels[group_id][round]);
    }

    function get_aggregater(uint group_id) public view returns(uint) {

        return groups[group_id].aggregater_id;
    }

    function vote(uint res, uint group_id, uint round, uint voters) public {

        vote_result[group_id][round] += res;
        vote_count[group_id][round] += 1;

        if (vote_count[group_id][round] == voters) {

            if (vote_result[group_id][round] >= (voters / 2)) {
                emit global_accept(round, group_id);
            }

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
    
    function get_G_count(uint group_id, uint round) public view returns (uint) {
        return modelG_count[group_id][round];
    }

    function get_D_count(uint group_id, uint round) public view returns (uint) {
        return modelD_count[group_id][round];
    }

}