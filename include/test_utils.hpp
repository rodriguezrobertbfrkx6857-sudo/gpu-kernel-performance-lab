#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gpu_lab::test {

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void require_near(float actual, float expected, float atol, float rtol, const std::string& name) {
    const float error = std::fabs(actual - expected);
    const float bound = atol + rtol * std::fabs(expected);
    require(error <= bound, name + " mismatch: actual=" + std::to_string(actual) +
                                " expected=" + std::to_string(expected));
}

template <typename T>
inline void require_array_near(const T* actual, const T* expected, std::size_t count,
                               double atol, double rtol, const std::string& name) {
    for (std::size_t i = 0; i < count; ++i) {
        const double a = static_cast<double>(actual[i]);
        const double b = static_cast<double>(expected[i]);
        if (std::fabs(a - b) > atol + rtol * std::fabs(b)) {
            throw std::runtime_error(name + " mismatch at index " + std::to_string(i));
        }
    }
}

inline void pass(const std::string& name) {
    std::cout << "PASS " << name << '\n';
}

}  // namespace gpu_lab::test

