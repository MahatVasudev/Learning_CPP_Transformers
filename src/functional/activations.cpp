#include "functional/activations.h"
#include "functional/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
namespace functional {
void ReLU::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "ReLU::forward", "X");
  require_contiguous(Y, "ReLU::forward", "Y");
  assert(Y.shape() == X.shape());
  assert(&X != &Y);
  const float *x = X.data();
  float *y = Y.data();
  std::size_t n = X.numel();
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

void ReLU::backward(const Tensor &X, const Tensor &Y, const Tensor &dY,
                    Tensor &dX, bool accumulate) const {
  require_contiguous(X, "ReLU::backward", "X");
  require_contiguous(Y, "ReLU::backward", "Y");
  require_contiguous(dX, "ReLU::backward", "dX");
  require_contiguous(dY, "ReLU::backward", "dY");
  assert(X.shape() == Y.shape());
  assert(Y.shape() == dY.shape());
  assert(X.shape() == dX.shape());

  assert(&dX != &dY);
  assert(&dX != &Y);
  assert(&dX != &X);

  const float *x = X.data();
  float *dx = dX.data();
  const float *dy = dY.data();

  std::size_t n = X.numel();
  for (std::size_t i = 0; i < n; ++i) {
    float g = x[i] > 0 ? dy[i] : 0;

    if (accumulate) {
      dx[i] += g;
    } else {
      dx[i] = g;
    }
  }
}

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

} // namespace functional
