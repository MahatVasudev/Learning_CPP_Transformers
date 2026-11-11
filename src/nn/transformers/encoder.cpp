
#include "core/errors.h"
#include "core/random.h"
#include "core/tensor.h"
#include "core/utils.h"
#include "nn/module.h"
#include "nn/utils.h"
#include <nn/transformers/encoders.h>
#include <vector>

using core::Tensor;
namespace error = core::error;
using std::size_t;

// Encoder Unit
// Represents one encoder unit
// mha -> ff

namespace nn::transformers {

EncoderUnit::EncoderUnit(size_t d_model, size_t d_ff, size_t heads,
                         core::random::Generator *generator)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model_, d_ff_, generator), mha_(d_model_, heads_, false, generator),
      ln_1(d_model), ln_2(d_model) {}

std::vector<Module *> EncoderUnit::modules() {
  return {&ln_1, &mha_, &ln_2, &ff_};
}

std::vector<optional<Tensor> *> EncoderUnit::buffers_() {
  return {&mha_output_, &norm_1_output_, &norm_2_output_};
}

std::vector<optional<Tensor> *> EncoderUnit::buffer_gradients_() {
  return {&d_mha_output_, &d_norm_1_output_, &d_norm_2_output_, &dX_,
          &dSource_};
}

// A single EncoderUnit forward
// uses PreNormalization i.e
// X + Sublayer(LayerNorm(X))
void EncoderUnit::forward(const Tensor &X, Tensor &Y, const Tensor *mask) {
  const size_t &rank_ = X.rank();

  const std::vector<size_t> &x_shape = X.shape();
  const std::vector<size_t> &y_shape = Y.shape();

  if (rank_ != 3) {
    throw error::ShapeError(
        "EncoderUnit::forward", "expected the shape of X to be in 3 dimensions",
        x_shape, error::make_message("received dimensions: ", rank_));
  }

  if (x_shape != y_shape) {
    throw error::ShapeError("EncoderUnit::forward",
                            "The shape of X and Y must be the same", x_shape,
                            error::make_message("received Y shape: ",
                                                error::format_shape(y_shape)));
  }

  if (x_shape[rank_ - 1] != d_model_) {
    throw error::ShapeError(
        "EncoderUnit::forward",
        "The last dimension of X or Y must be equal to d_model", x_shape,
        error::make_message("current d_model: ", d_model_));
  }

  ensure_buffers(X);

  ln_1.forward(X, *norm_1_output_);
  mha_.forward(*norm_1_output_, *mha_output_, mask);

  mha_output_->add(X);

  ln_2.forward(*mha_output_, *norm_2_output_);
  ff_.forward(*norm_2_output_, Y);

  Y.add(*mha_output_);
}

void EncoderUnit::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                           bool accumulate) {
  require_training("EncoderUnit::backward");
  if (!mha_output_ || mha_output_->shape() != X.shape()) {
    throw error::StateError("EncoderUnit::backward: call forward() first");
  }
  ensure_gradients(X);
  ff_.backward(*norm_2_output_, dY, *d_norm_2_output_, false);

  ln_2.backward(*mha_output_, *d_norm_2_output_, *d_mha_output_, false);

  d_mha_output_->add(dY);

  mha_.backward(*d_mha_output_, *norm_1_output_, *norm_1_output_, *dX_,
                *dSource_, false);
  dX_->add(*dSource_);
  ln_1.backward(X, *dX_, *d_norm_1_output_, false);

  if (accumulate) {
    dX.add(*d_mha_output_);
    dX.add(*d_norm_1_output_);
  } else {
    dX = *d_mha_output_;
    dX.add(*d_norm_1_output_);
  }
}

void EncoderUnit::ensure_buffers(const Tensor &X) {
  for (optional<Tensor> *buff : buffers_()) {
    if (buff) // IMP: TN 001
      utils::ensure(*buff, X.shape());
  }
}

void EncoderUnit::ensure_gradients(const Tensor &X) {
  for (optional<Tensor> *buff : buffer_gradients_()) {
    if (buff) // IMP: TN 001
      utils::ensure(*buff, X.shape());
  }
}

// IMPLEMENTING ENCODER LAYER

EncoderLayer::EncoderLayer(size_t num_layers, size_t d_model, size_t d_ff,
                           size_t heads, core::random::Generator *generator)
    : num_encoders_(num_layers), d_model_(d_model), d_ff_(d_ff), heads_(heads) {
  for (size_t i = 0; i < num_encoders_; i++) {
    encoders_.insert(encoders_.begin() + i,
                     EncoderUnit(d_model_, d_ff_, heads_, generator));
  }

  for (size_t i = 0; i < num_encoders_; i++) {
    module_ptrs_.insert(module_ptrs_.begin() + i, &encoders_[i]);
  }
}

std::vector<Module *> EncoderLayer::modules() { return module_ptrs_; }

std::vector<optional<Tensor> *> EncoderLayer::buffer_gradients_() {
  return {&grad_a_, &grad_b_};
};

void EncoderLayer::ensure_buffers(const Tensor &X) {
  if (num_encoders_ == 0) {
    return;
  }

  activations_.reserve(num_encoders_ - 1);
  for (size_t i = 0; i < num_encoders_ - 1; i++) {
    if (i >= activations_.size()) {
      activations_.emplace_back(X.shape());
    } else if (activations_[i].shape() != X.shape()) {
      activations_[i] = Tensor(X.shape());
    }
  }

  // activations_.resize(num_encoders_ - 1);
}

void EncoderLayer::ensure_gradients(const Tensor &X) {
  utils::ensure(grad_a_, X.shape());
  utils::ensure(grad_b_, X.shape());
}

void EncoderLayer::clear_local_cache() {
  Module::clear_local_cache();

  activations_.clear();
}

void EncoderLayer::forward(const Tensor &X, Tensor &Y, const Tensor *mask) {
  core::utils::require_nonoverlap(X, Y, "EncoderLayer::forward", "X", "Y");
  ensure_buffers(X);
  if (num_encoders_ == 0) {
    Y = X;
    return;
  }

  const Tensor *input = &X;
  for (size_t i = 0; i < num_encoders_; i++) {
    Tensor *output = (i == num_encoders_ - 1) ? &Y : &activations_[i];

    encoders_[i].forward(*input, *output, mask);

    input = output;
  }
}

void EncoderLayer::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                            bool accumulate) {
  Module::require_training("EncoderLayer::backward");
  core::utils::require_nonoverlap(X, dX, "Encoder::backward", "X", "dX");
  core::utils::require_nonoverlap(dY, dX, "Encoder::backward", "dY", "dX");

  if (num_encoders_ == 0) {
    if (accumulate) {
      dX.add(dY);
    } else {
      dX = dY;
    }
    return;
  }

  if (activations_.empty() || activations_.size() != num_encoders_ - 1 ||
      activations_[0].shape() != X.shape()) {
    throw error::StateError("EncoderLayer::backward: run forward() first");
  }

  const Tensor *current_grad = &dY;
  for (size_t i = num_encoders_; i-- > 0;) {
    const Tensor &input = (i == 0) ? X : this->activations_[i - 1];

    Tensor *next_grad;

    if (i == 0) {
      next_grad = &dX;
    } else {
      next_grad = (i % 2 == 0) ? &*grad_a_ : &*grad_b_;
    }

    encoders_[i].backward(input, *current_grad, *next_grad,
                          i == 0 ? accumulate : false);

    current_grad = next_grad;
  }
}

} // namespace nn::transformers
