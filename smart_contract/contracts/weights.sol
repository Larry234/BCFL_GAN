// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(uint => Item[]) public items;
    mapping(uint => Group) groups;

    int group_counts;

    uint[] group_ids;
    string[] global_models;

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
    event aggregater_selected(address aggregater);
    event fetch_global(string global_hs);
    event stopTraining();

    constructor() {
        group_counts = 0;
    }

    function init_group(uint MaxRegistry, uint group_id) public {

        address[] memory members = new address[](MaxRegistry);
        members[0] = msg.sender;
        Group memory initial_group = Group(1, 0, members);
        groups[group_id] = initial_group;
        group_ids.push(group_id);
        group_counts++;
    }

    function join_group(uint group_id) public {

        groups[group_id].member_count += 1;
        groups[group_id].members.push(msg.sender); // add new member into address array

        if (groups[group_id].member_count == groups[group_id].members.length) {
            emit startTraining(group_id);
        }
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

    function upload_model(string memory hs, uint round, uint group_id) public {

        Item memory item = Item(msg.sender, hs);
        items[round].push(item);

        // check if all clients upload their model
        if (items[round].length == groups[group_id].member_count) {
            emit allModel_uploaded();
        }
    }

    function upload_global_model(string memory hs, uint round, uint group_id) public {

        uint selected = groups[group_id].aggregater_id;
        require(groups[group_id].members[selected] == msg.sender, "You are not the selected aggregater this round!");

        global_models[round] = hs; // upload global model

        // update aggregater for next round
        groups[group_id].aggregater_id = (groups[group_id].aggregater_id + 1) % groups[group_id].member_count;

        // trigger event to let client update model
        emit fetch_global(hs);
        
    }


    function fetch_model(uint round, uint group_id) public view returns(string [] memory) {

        uint data_count = groups[group_id].member_count;
        string[] memory ret = new string[](data_count);

        // fetch all model stored in contract
        for (uint i = 0; i < data_count; i++) {
            ret[i] = items[round][i].IPFS_hash;
        }
        return ret;
    }

    function fetch_global_model(uint round) public view returns (string memory) {

        require(keccak256(bytes(global_models[round])) != keccak256(bytes("")), 'Global Model not upload yet!');
        return global_models[round];
    }


    function get_aggregater(uint group_id) public view returns(uint) {

        return groups[group_id].aggregater_id;
    }


    // function vote() public {

    // }

    // function validate() public {

    // }

}