#include <cmath>
#include <core/tensor.h>
#include <core/utils.h>
#include <nn/activations/softmax.h>
using core::Tensor;

using core::utils::require_contiguous;

namespace nn::activations {

void Softmax::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "Softmax::forward", "X");
  require_contiguous(Y, "Softmax::forward", "Y");
  assert(X.shape() == Y.shape());
  assert(&X != &Y);
  assert(dim_ == -1);

  const std::size_t rows = X.numel() / X.shape().back();
  const std::size_t cols = X.shape().back();

  const float *x = X.data();
  float *y = Y.data();

  for (std::size_t row = 0; row < rows; ++row) {
    const float *x_row = x + row * cols;
    float *y_row = y + row * cols;

    float max_value = x_row[0];

    for (std::size_t j = 1; j < cols; ++j) {
      max_value = std::max(max_value, x_row[j]);
    }

    float sum = 0.0f;
    for (std::size_t j = 0; j < cols; ++j) {
      y_row[j] = std::exp(x_row[j] - max_value);
      sum += y_row[j];
    }

    for (std::size_t j = 0; j < cols; ++j) {
      y_row[j] /= sum;
    }
  }
}

void Softmax::backward(const Tensor &X, const Tensor &Y, const Tensor &dY,
                       Tensor &dX, bool accumulate) const {

  require_contiguous(X, "Softmax::backward", "X");
  require_contiguous(Y, "Softmax::backward", "Y");
  require_contiguous(dX, "Softmax::backward", "dX");
  require_contiguous(dY, "Softmax::backward", "dY");
  assert(X.shape() == Y.shape());
  assert(Y.shape() == dY.shape());
  assert(X.shape() == dX.shape());

  assert(dim_ == -1);

  assert(&dX != &dY);
  assert(&dX != &Y);
  assert(&dX != &X);

  const std::size_t rows = X.numel() / X.shape().back();
  const std::size_t cols = X.shape().back();

  const float *y = Y.data();
  float *dx = dX.data();
  const float *dy = dY.data();

  for (std::size_t row = 0; row < rows; ++row) {
    const float *y_row = y + row * cols;
    const float *dy_row = dy + row * cols;
    float *dx_row = dx + row * cols;

    float dot = 0.0f;

    for (std::size_t j = 0; j < cols; ++j) {
      dot += dy_row[j] * y_row[j];
    }

    for (std::size_t j = 0; j < cols; ++j) {
      const float g = y_row[j] * (dy_row[j] - dot);

      if (accumulate) {
        dx_row[j] += g;
      } else {
        dx_row[j] = g;
      }
    }
  }
}
} // namespace nn::activations
