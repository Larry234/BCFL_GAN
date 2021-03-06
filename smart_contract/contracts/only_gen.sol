// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(uint => Group) groups;

    uint[50][8] modelG_count;
    string[][50][8] gen_models;
    string[50][8] global_genModels;
    uint[][50][8] validators;
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
        groups[group_id].members.push(msg.sender);
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

    function upload_model(string memory gen, uint round, uint group_id) public {

        require(modelG_count[group_id][round] < groups[group_id].member_count, "too much generator models uploaded!");
        modelG_count[group_id][round] += 1;
        gen_models[group_id][round].push(gen);

        // check if all clients upload their model
        if (modelG_count[group_id][round] == groups[group_id].member_count) 
        {
            emit allModel_uploaded(round, group_id);
        }

    }

    function upload_globalModel(string memory gen, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        string memory empty = "";

        // only chosen aggregater can upload global model
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!"); 

        // global model cannot be uploaded more than once
        require(keccak256(abi.encodePacked(global_genModels[group_id][round])) == keccak256(abi.encodePacked(empty)), "Cannot upload global model more than once!");
        global_genModels[group_id][round] = gen; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        emit aggregation_complete(round, group_id);
        
    }

    function fetch_model(uint round, uint group_id) public view returns(string [] memory) {

        string[] memory gen_ret = new string[](modelG_count[group_id][round]);

        for (uint i = 0; i < gen_ret.length; i++) {
            gen_ret[i] = gen_models[group_id][round][i];
        }

        return gen_ret;
    }

    function fetch_global_model(uint round, uint group_id) public view returns (string memory) {

        return global_genModels[group_id][round];
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

    function choose_validator(uint group_id, uint round, uint[] memory vals) public {
        
        for (uint i = 0; i < vals.length; i++) {
            validators[group_id][round].push(vals[i]);
        }
    }

    function get_validator(uint group_id, uint round) public view returns (uint[] memory) {

        return validators[group_id][round];
    }


    // ============================================================================
    // utility functions for debugging

    function get_genCount(uint round) public view returns (uint) {
        return gen_models[round].length;
    }

    function get_member_count(uint group_id) public view returns (uint) {
        return groups[group_id].member_count;
    }
    
    function get_G_count(uint group_id, uint round) public view returns (uint) {
        return modelG_count[group_id][round];
    }

    function get_G_model(uint group_id, uint round) public view returns (string memory){
        return global_genModels[group_id][round];
    }
}