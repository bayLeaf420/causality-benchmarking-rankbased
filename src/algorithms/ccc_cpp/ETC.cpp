#include <iostream>
#include <vector>
#include <utility>
#include <cstdint>

class ETC_Node
{
/*
Custom class made to store a 'node' used during ETC iterations.
*/
private:
    uint16_t hash_value;
    std::pair<uint16_t, uint16_t> indices;

public:
    // Functions to access hash and indices
    uint16_t get_hash()
    {
        return hash_value;
    }
    std::pair<uint16_t, uint16_t> get_indices()
    {
        return indices;
    }

    // Functions to set hash and indices
    void set_hash(uint16_t val)
    {
        hash_value = val;
        return;
    }
    void set_indices(uint16_t val1, uint16_t val2)
    {
        indices.first = val1;
        indices.second = val2;
        return;
    }
};