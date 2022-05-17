// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract NetworkWeights {
    
    mapping(address => Item[]) public items;
    mapping(uint => Group) groups;

    uint[] group_ids;

    // data structures
    struct Item {
        uint8 round;
        string IPFS_hash;
    }

    struct Group {
        uint8 member_count;
        address[] members;
    }
    
    // events
    event aggregater_selected(address aggregater);
    event startNextIter(); // signals to inform client nodes to start next iteration
    event stopTraining();

    constructor() {

    }

    function init_group(uint MaxRegistry, uint group_id) public {

        address[] memory members = new address[](MaxRegistry);
        members[0] = msg.sender;
        Group memory initial_group = Group(1, members);
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

    function upload_model(string memory hs) public {

    }

    function fetch_model() public view returns(string memory) {

    }

    function select_aggregater() public returns(uint8) {

    }

    function validate() public {

    }



}