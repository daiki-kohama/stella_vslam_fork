#ifndef STELLA_VSLAM_UTIL_RANDOM_ARRAY_H
#define STELLA_VSLAM_UTIL_RANDOM_ARRAY_H

#include <vector>
#include <random>
#include <memory>
#include "stella_vslam/util/random_seed.h"

namespace stella_vslam {
namespace util {

// Create random_engine.
// If random_seed has a value, that seed is used regardless of use_fixed_seed.
// If use_fixed_seed is true (and random_seed is not set), a fixed default seed is used.
// Otherwise, a random seed from std::random_device is used.
std::mt19937 create_random_engine(bool use_fixed_seed = false,
                                  random_seed_t random_seed = random_seed_t{});

template<typename T>
std::vector<T> create_random_array(const size_t size, const T rand_min, const T rand_max,
                                   std::mt19937& random_engine);

} // namespace util
} // namespace stella_vslam

#endif // STELLA_VSLAM_UTIL_RANDOM_ARRAY_H
