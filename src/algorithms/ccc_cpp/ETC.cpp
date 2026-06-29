#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <stack>

/*
We store 3 things:
    1. original array where we insert -1s for numbers removed during pair compression.
    2. std::multimap where the key_value = hash and values = indice pairs.
*/
class Dynamic_Node
{
private:
    // std::vector<int> sequence;
    // std::vector<int> hashed_pair_sequence;
    int priority_value;
    std::stack<std::pair<int, int>> indices_list;

public:
    int _get_priority_value(){
        return priority_value;
    }
    

    int _initialise(int priority_value, std::pair<int, int> indices){

    }
};

class Dynamic_Symbolic_Sequence{
    /*
    Here I will implement the following things:

    1. A function to construct the hash vector, and the BST Tree consisting of DynamicNodes. 
    2. A function to substitute the most frequently occuring pair in the sequence with a new one. 

    This is all, because a) Binning, b) Tracking of bins, and c) Passing in max_length, will be done 
    inside the ETC function. 

    This will include data structures:
    1. A vector which is of shape (max_hash_value,) where at the indice [hash1] there will be a pointer or
    indice to the Dynamic Node in the BST Tree corresponding to that hash value. 
    2. The std::set containing DynamicNode objects

    */
private:
    vector<int>
   
};
