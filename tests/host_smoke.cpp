#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

int main() {
    using gpu_lab::test::pass;
    using gpu_lab::test::require;
    constexpr std::size_t count = 4096;
    std::vector<float> a(count), b(count), c(count);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = static_cast<float>(count - i) * 0.125f;
        c[i] = a[i] + b[i];
    }
    for (std::size_t i = 0; i < count; ++i) {
        require(std::fabs(c[i] - (a[i] + b[i])) < 1.0e-6f, "vector add reference");
    }
    pass("cpu vector-add reference");

    const std::size_t rows = 37;
    const std::size_t cols = 29;
    std::vector<float> matrix(rows * cols), transposed(cols * rows);
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = static_cast<float>(i);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            transposed[col * rows + row] = matrix[row * cols + col];
        }
    }
    require(transposed[7 * rows + 11] == matrix[11 * cols + 7], "transpose reference");
    pass("cpu transpose reference");

    const float sum = std::accumulate(a.begin(), a.end(), 0.0f);
    require(sum > 0.0f, "reduction reference");
    pass("cpu reduction reference");

    constexpr int m = 17;
    constexpr int n = 13;
    constexpr int k = 11;
    std::vector<float> left(m * k, 1.0f), right(k * n, 2.0f), product(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                product[row * n + col] += left[row * k + inner] * right[inner * n + col];
            }
            require(product[row * n + col] == static_cast<float>(2 * k), "gemm reference");
        }
    }
    pass("cpu gemm reference");
    return 0;
}

