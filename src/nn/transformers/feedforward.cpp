#include "core/random.h"
#include "nn/utils.h"
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

void FeedForward::ensure_buffers_(const Tensor &X) {
  for (std::optional<Tensor> *buff : buffers_()) {
    if (buff) // IMP: TN 001
      nn::utils::ensure(*buff, X.shape());
  }
}

void FeedForward::ensure_gradients_(const Tensor &X) {
  for (std::optional<Tensor> *buff : buffer_gradients_()) {
    if (buff) // IMP: TN 001
      nn::utils::ensure(*buff, X.shape());
  }
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

void FeedForward::forward(const Tensor &X, Tensor &Y) {
  if (Y.shape() != X.shape())
    throw error::ShapeError("FeedForward::forward",
                            "the shape of Y must be equal to X", Y.shape(),
                            error::make_message("the shape of X is ",
                                                error::format_shape(X.shape()))

    );
  core::utils::require_nonoverlap(X, Y, "FeedForward::forward", "X", "Y");

  ensure_buffers_(X);

  l1_.forward(X, *h_pre_);
  relu_.forward(*h_pre_, *h_act_);
  l2_.forward(*h_act_, Y);
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

std::vector<Module *> FeedForward::modules() { return {&l1_, &l2_}; }

std::vector<std::optional<Tensor> *> FeedForward::buffers_() {
  return {&h_pre_, &h_act_};
}

std::vector<std::optional<Tensor> *> FeedForward::buffer_gradients_() {
  return {&dh_pre_, &dh_act_};
};
} // namespace nn::transformers
