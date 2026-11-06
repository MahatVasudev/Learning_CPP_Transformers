#ifndef UTILS_NN_H

#include <core/tensor.h>
#include <cstddef>
#include <optional>

using core::Tensor;

#define UTILS_NN_H

namespace nn::utils {

double angle(std::size_t pos, std::size_t i, std::size_t d_model);

void ensure(std::optional<Tensor> &buffer,
            const std::vector<std::size_t> &shape);

template <typename T> T iterative_product_vector(std::vector<T> &data);
} // namespace nn::utils

#endif // !UTILS_NN_H
