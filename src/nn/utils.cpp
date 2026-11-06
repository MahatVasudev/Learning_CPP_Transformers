#include <cmath>
#include <core/tensor.h>
#include <cstddef>
#include <nn/utils.h>
#include <optional>
#include <vector>

using core::Tensor;

namespace nn::utils {

double angle(std::size_t pos, std::size_t i, std::size_t d_model) {

  const double exponent =
      2.0 * (static_cast<double>(i) / 2.0) / static_cast<double>(d_model);

  return static_cast<double>(pos) / std::pow(10000.0, exponent);
}

void ensure(std::optional<Tensor> &buffer,
            const std::vector<std::size_t> &shape) {

  if (buffer && buffer->shape() == shape) {
    return;
  }

  buffer.emplace(shape);
}

template <typename T> T iterative_product_vector(std::vector<T> &data) {
  assert(std::is_arithmetic_v<T>);
  T value = static_cast<T>(1);

  for (T v : data) {
    value *= v;
  }

  return value;
}

} // namespace nn::utils
