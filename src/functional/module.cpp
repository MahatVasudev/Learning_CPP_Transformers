#include <algorithm>
#include <cmath>
#include <cstdint>
#include <errors/errors.h>
#include <functional/module.h>
#include <functional/ops.h>
#include <functional/tensor.h>
#include <optional>
#include <utils/utils.h>
#include <vector>

namespace functional {

std::vector<Tensor *>
Module::concat_parameters(const std::vector<Module *> &modules) {
  std::vector<Tensor *> global_vec;
  for (Module *mod : modules) {
    assert(mod != nullptr);
    const std::vector<Tensor *> p = mod->parameters();
    global_vec.insert(global_vec.end(), p.begin(), p.end());
  }

  return global_vec;
}

std::vector<Tensor *>
Module::concat_gradients(const std::vector<Module *> &modules) {
  std::vector<Tensor *> global_vec;
  for (Module *mod : modules) {
    assert(mod != nullptr);
    const std::vector<Tensor *> g = mod->gradients();
    global_vec.insert(global_vec.end(), g.begin(), g.end());
  }

  return global_vec;
}

void Module::set_training_all(const std::vector<Module *> &modules, bool mode) {
  for (Module *mod : modules) {
    assert(mod != nullptr);
    mod->set_training(mode);
  }
}

void LayerNorm::set_training(bool state) {
  Module::set_training(state);

  if (!state) {
    gamma_grad_.reset();
    beta_grad_.reset();
  }
}

void LayerNorm::ensure_gradients_() {
  if (!gamma_grad_)
    gamma_grad_.emplace(std::vector<std::size_t>{normalized_shape_});
  if (!beta_grad_)
    beta_grad_.emplace(std::vector<std::size_t>{normalized_shape_});
}

void LayerNorm::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "LayerNorm::forward", "X");
  require_contiguous(Y, "LayerNorm::forward", "Y");
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

void LayerNorm::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                         bool accumulate) {
  if (!training_) {
    throw error::StateError("LayerNorm::backward: cannot run backward() on "
                            "eval(), run train() first");
  }
  require_contiguous(X, "LayerNorm::backward", "X");
  require_contiguous(dY, "LayerNorm::backward", "dY");
  require_contiguous(dX, "LayerNorm::backward", "dX");
  assert(X.rank() >= 1);
  assert(X.shape() == dX.shape());
  assert(X.shape() == dY.shape());
  assert(X.shape().back() == normalized_shape_);
  assert(&X != &dX);
  assert(&dX != &dY);

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

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               unsigned seed)
    : in_(in_features), out_(out_features), bias_(bias),
      W_(Tensor({out_features, in_features})), B_(Tensor({out_features})) {
  assert(in_features > 0 && out_features > 0);
  utils::fill_random(W_, utils::Initialize::xavier_uniform(), seed);
}

void Linear::set_training(bool state) {
  Module::set_training(state);
  if (!state) {
    dW_.reset();
    dB_.reset();
  }
}

void Linear::ensure_gradients_() {
  if (!dW_)
    dW_.emplace(std::vector<std::size_t>{out_, in_});
  if (!dB_ && bias_)
    dB_.emplace(std::vector<std::size_t>{out_});
}

void Linear::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "Linear::forward", "X");
  require_contiguous(Y, "Linear::forward", "Y");
  assert(X.rank() >= 1 && X.shape().back() == in_);
  assert(Y.rank() == X.rank() && Y.shape().back() == out_);

  for (std::size_t d = 0; d + 1 < X.rank(); ++d) {
    assert(X.shape()[d] == Y.shape()[d]);
  }
  assert(&X != &Y);
  Y.fill(0.0f);
  if (bias_) {
    Y.add(B_);
  }

  matmul(X, W_, Y, false, true, true);
}

void Linear::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                      bool accumulate) {

  if (!training_) {
    throw error::StateError("Linear::backward: cannot run backward() on "
                            "eval(), run train() first");
  }
  require_contiguous(X, "Linear::backward", "X");
  require_contiguous(dY, "Linear::backward", "dY");
  require_contiguous(dX, "Linear::backward", "dX");
  assert(X.rank() >= 1 && X.shape().back() == in_);
  assert(dY.shape().back() == out_);
  assert(dX.shape() == X.shape());
  assert(dY.numel() / out_ == X.numel() / in_);
  assert(&dX != &X && &dX != &dY);
  ensure_gradients_();

  matmul(dY, W_, dX, false, false, accumulate);
  // gemm(dY.data(), W_.data(), dX.data(), rows, out_, in_, out_, 1, in_, 1,
  // in_,
  //      1, false, false, accumulate);
  const std::size_t rows = X.numel() / in_;
  Tensor X2 = X.view({rows, in_});
  Tensor dY2 = dY.view({rows, out_});

  matmul(dY2, X2, *dW_, true, false, true);

  if (bias_) {
    // const float *dy = dY.data();
    // float *db = dB_->data();
    // for (std::size_t r = 0; r < rows; ++r)
    //   for (std::size_t j = 0; j < out_; ++j)
    //     db[j] += dy[r * out_ + j];
    // std::cout << error::format_shape(dB_->shape())
    //           << error::format_shape(dY.shape());
    dB_->add(dY2.sum(0));
  }
}

std::vector<Tensor *> Linear::parameters() {
  if (bias_) {
    return {&W_, &B_};
  }
  return {&W_};
}

std::vector<Tensor *> Linear::gradients() {
  ensure_gradients_();
  if (bias_) {
    return {&*dW_, &*dB_};
  }

  return {&*dW_};
}

Embedding::Embedding(std::size_t vocab_size, std::size_t d_model, unsigned seed,
                     std::optional<std::size_t> padding_idx)
    : vocab_size_(vocab_size), d_model_(d_model), padding_idx_(padding_idx),
      Emb_({vocab_size, d_model}) {
  assert(vocab_size > 0 && d_model > 0);
  if (padding_idx_ && *padding_idx_ >= vocab_size_) {
    throw error::IndexError(
        "Embedding: padding_idx is out of bounds", *padding_idx_,
        error::make_message("padding_idx cannot be greater than ", vocab_size_,
                            " (vocab_size)"));
  }
  utils::fill_random(Emb_, utils::Initialize::normal(0.0f, 1.0f), seed);

  if (padding_idx_) {
    float *row = Emb_.data() + *padding_idx_ * d_model_;
    std::fill(row, row + d_model_, 0.0f);
  }
}

std::size_t Embedding::checked_id_(std::int32_t id) const {
  std::size_t _id = static_cast<std::size_t>(id);
  if (id < 0 || _id >= vocab_size_) {
    throw error::IndexError(
        "Embedding id is out of range", _id,
        error::make_message("id cannot be less than 0 or greater than ",
                            vocab_size_, " (vocab_size)"));
  }

  return _id;
}

void Embedding::set_training(bool state) {
  Module::set_training(state);

  dEmb_.reset();
}

void Embedding::ensure_gradient() {
  if (!dEmb_)
    dEmb_.emplace({vocab_size_, d_model_});
}

void Embedding::forward(const std::int32_t *ids, std::size_t n,
                        Tensor &Y) const {
  require_contiguous(Y, "Embedding::forward", "Y");
  assert(ids != nullptr);
  assert(Y.rank() >= 1 && Y.shape().back() == d_model_);
  assert(Y.numel() == n * d_model_);
  assert(&Y != &Emb_);

  const float *emb = Emb_.data();
  float *y = Y.data();

  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t id = checked_id_(ids[i]);
    const float *src = emb + id * d_model_;
    float *dst = y + i * d_model_;
    std::copy(src, src + d_model_, dst);
  }
}

void Embedding::backward(const std::int32_t *ids, std::size_t n,
                         const Tensor &dY) {

  require_contiguous(dY, "Embedding::forward", "Y");
  if (!training_) {
    throw error::StateError("Embedding::backward: cannot run backward() on "
                            "eval(), run train() first");
  }
  assert(ids != nullptr);
  assert(dY.rank() >= 1 && dY.shape().back() == d_model_);
  assert(dY.numel() == n * d_model_);

  ensure_gradient();
  const float *dy = dY.data();
  float *grad = dEmb_->data();

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t id = checked_id_(ids[i]);
    if (id == padding_idx_)
      continue;
    const float *src = dy + i * d_model_;
    float *dst = grad + id * d_model_;
    for (std::size_t j = 0; j < d_model_; ++j) {
      dst[j] += src[j];
    }
  }
}

std::vector<Tensor *> Embedding::parameters() { return {&Emb_}; }

std::vector<Tensor *> Embedding::gradients() {
  ensure_gradient();
  return {&*dEmb_};
}
} // namespace functional
