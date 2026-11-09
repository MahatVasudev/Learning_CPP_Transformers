#include "core/random.h"
#include <core/errors.h>
#include <core/tensor.h>
#include <core/utils.h>
#include <nn/transformers/feedforward.h>

using core::Tensor;

namespace error = core::error;

namespace nn::transformers {

// FeedForward Implementation -------------------------------

FeedForward::FeedForward(std::size_t d_model, std::size_t d_ff,
                         core::random::Generator *generator)
    : d_model_(d_model), d_ff_(d_ff),
      l1_(Linear(d_model, d_ff, true, generator)),
      l2_(Linear(d_ff, d_model, true, generator)), relu_() {}

void FeedForward::set_training(bool state) {
  Module::set_training(state);

  Module::set_training_all({&l1_, &l2_}, state);

  if (!state) {
    dh_pre_.reset();
    dh_act_.reset();
  }
}

void FeedForward::ensure_gradients_(const Tensor &X) {
  const std::vector<std::size_t> shape = this->hidden_shape_(X);
  if (dh_pre_ && dh_pre_->shape() == shape)
    return; // steady state: no allocation
  dh_pre_.emplace(shape);
  dh_act_.emplace(shape);
}

std::vector<std::size_t> FeedForward::hidden_shape_(const Tensor &X) const {
  if (X.rank() < 1 || X.shape().back() != d_model_)
    throw error::ShapeError(
        "FeedForward",
        error::make_message("last dimension of X must be equal to (d_model) ",
                            d_model_),
        X.shape());
  std::vector<std::size_t> shape = X.shape();
  shape.back() = d_ff_;
  return shape;
}

void FeedForward::ensure_buffers_(const Tensor &X) {
  const std::vector<std::size_t> shape = this->hidden_shape_(X);
  if (h_pre_ && h_pre_->shape() == shape)
    return; // steady state: no allocation
  h_pre_.emplace(shape);
  h_act_.emplace(shape);
}

void FeedForward::forward(const Tensor &X, Tensor &Y) {
  if (Y.shape() != X.shape())
    throw error::ShapeError("FeedForward::forward",
                            "the shape of Y must be equal to X", Y.shape(),
                            error::make_message("the shape of X is ",
                                                error::format_shape(X.shape()))

    );
  core::utils::require_nonoverlap(X, Y, "FeedForward::forward", "X", "Y");
  l1_.forward(X, *h_pre_);
  // std::cout << "pased l1";
  relu_.forward(*h_pre_, *h_act_);
  // std::cout << "pased l2";
  l2_.forward(*h_act_, Y);
  // std::cout << "pased l3";
}

void FeedForward::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                           bool accumulate) {
  require_training("FeedForward::backward");
  if (!h_pre_ || h_pre_->shape() != hidden_shape_(X))
    throw error::StateError("FeedForward::backward: call forward() first");
  if (dY.shape() != X.shape() || dX.shape() != X.shape())
    throw error::ShapeError(
        "FeedForward::backward", "X's shape must match with dY and dX",
        X.shape(),
        error::make_message("dX's shape: ", error::format_shape(dX.shape()),
                            "dY's shape: ", error::format_shape(dY.shape())));
  core::utils::require_nonoverlap(X, dX, "FeedForward::backward", "X", "dX");
  core::utils::require_nonoverlap(dY, dX, "FeedForward::backward", "dY", "dX");
  ensure_gradients_(X);

  l2_.backward(*h_act_, dY, *dh_act_, false);
  relu_.backward(*h_pre_, *dh_act_, *dh_pre_, false);
  l1_.backward(X, *dh_pre_, dX, accumulate);
}

std::vector<Tensor *> FeedForward::parameters() {
  return this->concat_parameters({&l1_, &l2_});
}

std::vector<Tensor *> FeedForward::gradients() {
  return this->concat_gradients({&l1_, &l2_});
}

} // namespace nn::transformers
