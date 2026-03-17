#ifndef STELLA_VSLAM_UTIL_RANDOM_SEED_H
#define STELLA_VSLAM_UTIL_RANDOM_SEED_H

#include <cassert>

namespace stella_vslam {
namespace util {

//! C++11-compatible type representing an optional random seed.
//! Used instead of std::optional<unsigned int> (which requires C++17).
struct random_seed_t {
    random_seed_t() : has_value_(false), value_(0) {}
    explicit random_seed_t(unsigned int v) : has_value_(true), value_(v) {}
    bool has_value() const { return has_value_; }
    unsigned int value() const { assert(has_value_); return value_; }

private:
    bool has_value_;
    unsigned int value_;
};

} // namespace util
} // namespace stella_vslam

#endif // STELLA_VSLAM_UTIL_RANDOM_SEED_H
