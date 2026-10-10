#include "functional/functional.hpp"
#include "functional/ops.h"
#include <cmath>
#include <random>
#include <vector>

namespace functional {
void LayerNorm::forward(const Tensor &X, Tensor &Y) {
  assert(X.rank() >= 1);
  assert(X.shape().back() == normalized_shape_);
  assert(Y.shape() == X.shape());
  assert(&X != &Y);

  const std::size_t D = normalized_shape_;
  const std::size_t rows = X.numel() / D;

  const float *x = X.data();
  float *y = Y.data();

  for (std::size_t row = 0; row < rows; ++row) {
    const float *x_row = x + row * D;
    float *y_row = y + row * D;

    float mean = mean_calculate_(x_row, D);

    float variance = variance_calculate_(x_row, mean, D);

    const float rstd = 1.0f / std::sqrt(variance + epsilon_);

    for (std::size_t j = 0; j < D; ++j) {
      const float normalized = (x_row[j] - mean) * rstd;
      y_row[j] = gamma_.data()[j] * normalized + beta_.data()[j];
    }
  }
}

void LayerNorm::backward(const Tensor &X, const Tensor &Y, const Tensor &dY,
                         Tensor &dX, bool accumulate) {

  assert(X.rank() >= 1);
  assert(X.shape() == Y.shape());
  assert(Y.shape() == dY.shape());
  assert(X.shape() == dX.shape());
  assert(X.shape().back() == normalized_shape_);
  assert(&X != &Y);
  assert(&Y != &dY);
  assert(&X != &dX);
  const std::size_t D = normalized_shape_;
  const std::size_t rows = X.numel() / D;

  const float *x = X.data();
  const float *dy = dY.data();
  float *dx = dX.data();

  for (std::size_t row = 0; row < rows; ++row) {
    const float *x_row = x + row * D;
    const float *dy_row = dy + row * D;
    float *dx_row = dx + row * D;

    float mean = mean_calculate_(x_row, D);
    float variance = variance_calculate_(x_row, mean, D);

    const float rstd = 1.0f / std::sqrt(variance + epsilon_);

    float sum_dy = 0.0f;
    float sum_dy_norm = 0.0f;

    for (std::size_t j = 0; j < D; ++j) {
      const float normalized = (x_row[j] - mean) * rstd;
      const float dnorm = dy_row[j] * gamma_.data()[j];

      sum_dy += dnorm;
      sum_dy_norm += dnorm * normalized;

      gamma_grad_.data()[j] += dy_row[j] * normalized;
      beta_grad_.data()[j] += dy_row[j];
    }

    for (std::size_t j = 0; j < D; ++j) {
      const float normalized = (x_row[j] - mean) * rstd;
      const float dnorm = dy_row[j] * gamma_.data()[j];

      float g = (dnorm - sum_dy / static_cast<float>(D) -
                 normalized * sum_dy_norm / static_cast<float>(D)) *
                rstd;

      if (accumulate) {
        dx_row[j] += g;
      } else {
        dx_row[j] = g;
      }
    }
  }
}

float LayerNorm::mean_calculate_(const float *x, std::size_t by) const {
  float mean = 0.0f;
  for (std::size_t j = 0; j < by; ++j) {
    mean += x[j];
  }

  mean /= static_cast<float>(by);

  return mean;
}

float LayerNorm::variance_calculate_(const float *x, const float mean,
                                     std::size_t by) const {
  float variance = 0.0f;

  for (std::size_t j = 0; j < by; ++j) {
    const float centered = x[j] - mean;
    variance += centered * centered;
  }

  variance /= static_cast<float>(by);

  return variance;
}

std::vector<Tensor *> LayerNorm::parameters() { return {&gamma_, &beta_}; }

std::vector<Tensor *> LayerNorm::gradients() {
  return {&gamma_grad_, &beta_grad_};
}

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               unsigned seed)
    : in_(in_features), out_(out_features), bias_(bias), seed_(seed),
      W_(Tensor({out_features, in_features})), B_(Tensor({out_features})),
      dW_(Tensor({out_features, in_features})), dB_(Tensor({out_features})) {
  assert(in_features > 0 && out_features > 0);

  std::mt19937 rng(seed);

  const float bound = std::sqrt(6.0f / static_cast<float>(in_ + out_));

  std::uniform_real_distribution<float> dist(-bound, bound);

  float *w = W_.data();

  for (std::size_t i = 0; i < W_.numel(); ++i) {
    w[i] = dist(rng);
  }
}

void Linear::forward(const Tensor &X, Tensor &Y) const {
  assert(X.rank() >= 1 && X.shape().back() == in_);
  assert(Y.rank() == X.rank() && Y.shape().back() == out_);

  for (std::size_t d = 0; d + 1 < X.rank(); ++d) {
    assert(X.shape()[d] == Y.shape()[d]);
  }
  assert(&X != &Y);

  const std::size_t rows = X.numel() / in_;

  float *y = Y.data();

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < out_; ++col) {
      y[row * out_ + col] = bias_ ? B_.data()[col] : 0.0f;
    }
  }

  gemm(X.data(), W_.data(), y, rows, in_, out_, in_, in_, out_, false, true,
       true);
}

void Linear::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                      bool accumulate) {

  assert(X.rank() >= 1 && X.shape().back() == in_);
  assert(dY.shape().back() == out_);
  assert(dX.shape() == X.shape());
  assert(dY.numel() / out_ == X.numel() / in_);
  assert(&dX != &X && &dX != &dY);

  const std::size_t rows = X.numel() / in_;

  gemm(dY.data(), W_.data(), dX.data(), rows, out_, in_, out_, in_, in_, false,
       false, accumulate);

  gemm(dY.data(), X.data(), dW_.data(), out_, rows, in_, out_, in_, in_, true,
       false, true);

  if (bias_) {
    const float *dy = dY.data();
    float *db = dB_.data();
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t j = 0; j < out_; ++j)
        db[j] += dy[r * out_ + j];
  }
}

std::vector<Tensor *> Linear::parameters() {
  if (bias_) {
    return {&W_, &B_};
  }
  return {&W_};
}

std::vector<Tensor *> Linear::gradients() {
  if (bias_) {
    return {&dW_, &dB_};
  }

  return {&dW_};
}

} // namespace functional
