#include "functional/activations.h"
#include <cstddef>
#include <functional/module.h>
#include <model/model.h>
#include <stdexcept>
#include <vector>

using functional::Tensor;

namespace model {

FeedForward::FeedForward(std::size_t d_model, std::size_t d_ff, unsigned seed)
    : d_model_(d_model), d_ff_(d_ff),
      l1_(functional::Linear(d_model, d_ff, true, seed)),
      l2_(functional::Linear(d_ff, d_model, true, seed + 1)), relu_() {}

std::vector<std::size_t>
FeedForward::hidden_shape_(const functional::Tensor &X) const {
  if (X.rank() < 1 || X.shape().back() != d_model_)
    throw std::invalid_argument(
        "FeedForward: last dimension of X must equal d_model");
  std::vector<std::size_t> shape = X.shape();
  shape.back() = d_ff_;
  return shape;
}

void FeedForward::ensure_buffers_(const functional::Tensor &X) {
  const std::vector<std::size_t> shape = this->hidden_shape_(X);
  if (h_pre_ && h_pre_->shape() == shape)
    return; // steady state: no allocation
  h_pre_.emplace(shape);
  h_act_.emplace(shape);
  dh_pre_.emplace(shape);
  dh_act_.emplace(shape);
}

void FeedForward::forward(const functional::Tensor &X, functional::Tensor &Y) {
  if (Y.shape() != X.shape())
    throw std::invalid_argument("FeedForward::forward: Y must match X's shape");
  ensure_buffers_(X);

  l1_.forward(X, *h_pre_);
  relu_.forward(*h_pre_, *h_act_);
  l2_.forward(*h_act_, Y);
}

void FeedForward::backward(const functional::Tensor &X,
                           const functional::Tensor &dY, functional::Tensor &dX,
                           bool accumulate) {

  if (!h_pre_ || h_pre_->shape() != hidden_shape_(X))
    throw std::logic_error(
        "FeedForward::backward: call forward() on this X first");
  if (dY.shape() != X.shape() || dX.shape() != X.shape())
    throw std::invalid_argument(
        "FeedForward::backward: dY and dX must match X's shape");

  l2_.backward(*h_act_, dY, *dh_act_, false);
  relu_.backward(*h_pre_, *h_act_, *dh_act_, *dh_pre_, false);
  l1_.backward(X, *dh_pre_, dX, accumulate);
}

std::vector<Tensor *> FeedForward::parameters() {
  return this->concat_parameters({&l1_, &l2_});
}

std::vector<Tensor *> FeedForward::gradients() {
  return this->concat_gradients({&l1_, &l2_});
}

} // namespace model
