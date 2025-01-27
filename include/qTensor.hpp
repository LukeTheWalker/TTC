#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <set>
#include <array>
#include "bitset.hpp"

#ifdef USE_FLOAT
using dtype = float;
#else
using dtype = double;
#endif

class QTensor
{
    public:
        // std::set<unsigned char> span;
        // std::vector<std::complex<dtype> > values;
        unsigned char * span = nullptr;
        std::complex<dtype> * values = nullptr;
        QTensor() {}

        void setValues(std::vector<std::complex<dtype>> values) 
        {
            this->values = (std::complex<dtype> *)malloc(std::pow(2, rank * 2) * sizeof(std::complex<dtype>));
            if (values.size() != std::pow(2, rank * 2))
            {
                std::cerr << "Error: the number of values is not consistent with the rank of the tensor" << std::endl;
                exit(1);
            }else{
                for (size_t i = 0 ; i < values.size(); i++)
                {
                    this->values[i] = values[i];
                }
            } 
        }

        std::complex<dtype> getValue(size_t index) { return values[index]; }

        void printValues(std::ostream& os = std::cout) const
        {
            if (values == nullptr)
            {
                os << "No values for this tensor";
                return;
            }

            for (size_t i = 0 ; i < std::pow(2, 2 * rank); i++)
            {
                if (i != 0 && (i % (1 << rank)) == 0)
                {
                    os << std::endl;
                }
                os << values[i] << ", ";
            }
        }

        size_t getValuesSize() const { return std::pow(2, 2 * rank); }

        size_t getRank() const { return rank; }
    public:
        size_t rank;
};