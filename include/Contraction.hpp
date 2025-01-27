#pragma once

#include <vector>
#include <string>

#include "qTensor.hpp"
#include <cstdint>

struct Contraction {
    uint32_t id;
    uint32_t programId;
    std::vector<unsigned char> span;
    uint32_t leftId;
    uint32_t rightId;
    Contraction* left;
    Contraction* right;
    std::string kind;
    QTensor data;
};
