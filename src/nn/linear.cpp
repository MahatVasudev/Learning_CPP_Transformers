#include <core/errors.h>
#include <core/tensor.h>
#include <core/utils.h>
#include <cstddef>
#include <nn/initialization.h>
#include <nn/linear.h>
#include <ops/matmul.h>

using core::Tensor;
using core::utils::require_contiguous;
using nn::initialize::Initialize;

namespace error = core::error;

namespace nn {

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               unsigned seed)
    : in_(in_features), out_(out_features), bias_(bias),
      W_(Tensor({out_features, in_features})), B_(Tensor({out_features})) {
  assert(in_features > 0 && out_features > 0);
  fill_random(W_, Initialize::xavier_uniform(), seed);
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
  core::utils::require_nonoverlap(X, Y, "Linear::forward", "X", "Y");
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

  ops::matmul(X, W_, Y, false, true, true);
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
  core::utils::require_nonoverlap(X, dX, "Linear::backward", "X", "dX");
  core::utils::require_nonoverlap(dY, dX, "Linear::backward", "dY", "dX");
  ensure_gradients_();

  ops::matmul(dY, W_, dX, false, false, accumulate);
  // gemm(dY.data(), W_.data(), dX.data(), rows, out_, in_, out_, 1, in_, 1,
  // in_,
  //      1, false, false, accumulate);
  const std::size_t rows = X.numel() / in_;
  Tensor X2 = X.view({rows, in_});
  Tensor dY2 = dY.view({rows, out_});

  ops::matmul(dY2, X2, *dW_, true, false, true);

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

} // namespace nn
