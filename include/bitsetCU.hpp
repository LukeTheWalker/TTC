#pragma once

namespace sycl_classes {
    class bitset {
    public:
        unsigned char data[8];

        // Constructor with generic type
        template<typename T>
        SYCL_EXTERNAL bitset(T value) {
            *reinterpret_cast<size_t*>(data) = static_cast<size_t>(value);
        }

        // Default constructor
        SYCL_EXTERNAL bitset() : bitset(0ull) {}

        // Set a bit at given index
        SYCL_EXTERNAL void set(size_t index, bool value) {
            size_t byte_index = index >> 3;    // Divide by 8
            size_t bit_index = index & 7;      // Modulo 8
            if (value) {
                data[byte_index] |= (1 << bit_index);
            } else {
                data[byte_index] &= ~(1 << bit_index);
            }
        }

        // Get bit value at given index
        SYCL_EXTERNAL bool get(size_t index) const {
            size_t byte_index = index >> 3;    // Divide by 8
            size_t bit_index = index & 7;      // Modulo 8
            return (data[byte_index] >> bit_index) & 1;
        }

        // Convert to unsigned long long
        SYCL_EXTERNAL unsigned long long to_ulong() const {
            return *reinterpret_cast<const unsigned long long*>(data);
        }

        // Bitwise OR operator
        SYCL_EXTERNAL bitset operator|(const bitset& other) const {
            return bitset(this->to_ulong() | other.to_ulong());
        }

        // Transfer bit from this bitset to another
        SYCL_EXTERNAL void transfer_bit(size_t index, bitset& other) {
            size_t byte_index = index >> 3;
            size_t bit_index = index & 7;
            if (get(index)) {
                other.data[byte_index] |= (1 << bit_index);
            } else {
                other.data[byte_index] &= ~(1 << bit_index);
            }
        }

        // XOR operation with a size_t value
        SYCL_EXTERNAL void xor_op(size_t other) {
            *reinterpret_cast<size_t*>(data) ^= other;
        }
    };
}