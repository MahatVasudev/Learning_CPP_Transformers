#include <cmath>
#include <core/errors.h>
#include <core/tensor.h>
#include <core/utils.h>
#include <nn/layer_norm.h>

using core::Tensor;
using core::utils::require_contiguous;

namespace error = core::error;

namespace nn {

void LayerNorm::set_training(bool state) { Module::set_training(state); }

void LayerNorm::ensure_gradients_() {
  if (!gamma_grad_)
    gamma_grad_.emplace(std::vector<std::size_t>{normalized_shape_});
  if (!beta_grad_)
    beta_grad_.emplace(std::vector<std::size_t>{normalized_shape_});
}

void LayerNorm::clear_cache() {
  gamma_grad_.reset();
  beta_grad_.reset();
}

void LayerNorm::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "LayerNorm::forward", "X");
  require_contiguous(Y, "LayerNorm::forward", "Y");
  core::utils::require_nonoverlap(X, Y, "LayerNorm::forward", "X", "Y");
  assert(X.rank() >= 1);
  assert(X.shape().back() == normalized_shape_);
  assert(Y.shape() == X.shape());

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

void LayerNorm::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                         bool accumulate) {
  if (!training_) {
    throw error::StateError("LayerNorm::backward: cannot run backward() on "
                            "eval(), run train() first");
  }
  require_contiguous(X, "LayerNorm::backward", "X");
  require_contiguous(dY, "LayerNorm::backward", "dY");
  require_contiguous(dX, "LayerNorm::backward", "dX");
  core::utils::require_nonoverlap(X, dX, "LayerNorm::backward", "X", "dX");
  core::utils::require_nonoverlap(dY, dX, "LayerNorm::backward", "dY", "dX");
  assert(X.rank() >= 1);
  assert(X.shape() == dX.shape());
  assert(X.shape() == dY.shape());
  assert(X.shape().back() == normalized_shape_);

  ensure_gradients_();
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

      gamma_grad_->data()[j] += dy_row[j] * normalized;
      beta_grad_->data()[j] += dy_row[j];
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

  ensure_gradients_();
  return {&*gamma_grad_, &*beta_grad_};
}
} // namespace nn
