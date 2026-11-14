#include "core/random.h"
#include "core/utils.h"
#include "nn/module.h"
#include <core/errors.h>
#include <core/tensor.h>
#include <cstddef>
#include <nn/transformers/decoders.h>
#include <nn/utils.h>

using std::size_t;
namespace error = core::error;
namespace core_utils = core::utils;
namespace nn::transformers {

DecoderUnit::DecoderUnit(size_t d_model, size_t d_ff, size_t heads,
                         core::random::Generator *generator)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model, d_ff, generator), ln_1(d_model), ln_2(d_model),
      ln_3(d_model), mha_self(d_model, heads, true, generator),
      mha_cross(d_model, heads, false, generator) {}

std::vector<Module *> DecoderUnit::modules() {
  return {&ln_1, &mha_self, &ln_2, &mha_cross, &ln_3, &ff_};
}

std::vector<optional<Tensor> *> DecoderUnit::buffers_() {
  return {&norm_1_output_, &norm_2_output_, &norm_3_output_, &mha_cross_output_,
          &mha_self_output_};
}
std::vector<optional<Tensor> *> DecoderUnit::buffer_gradients_() {

  return {&d_norm_query_,
          &d_norm_source_,
          &d_norm_2_output_,
          &d_norm_3_output_,
          &d_mha_cross_output_,
          &d_mha_self_output_,
          &dX_,
          &dSource_};
}

void DecoderUnit::ensure_buffers(const Tensor &X) {
  for (optional<Tensor> *buff : buffers_()) {
    if (buff) // IMP: TN 001
      utils::ensure(*buff, X.shape());
  }
}

void DecoderUnit::ensure_gradients(const Tensor &X) {
  for (optional<Tensor> *buff : buffer_gradients_()) {
    if (buff) // IMP: TN 001
      utils::ensure(*buff, X.shape());
  }
}

void DecoderUnit::clear_local_cache() {
  Module::clear_local_cache();
  source_ = nullptr;
}

void DecoderUnit::forward(const Tensor &X, Tensor &Y, const Tensor *mask,
                          const Tensor *context) {

  const size_t &rank_ = X.rank();

  const std::vector<size_t> &x_shape = X.shape();
  const std::vector<size_t> &y_shape = Y.shape();

  if (rank_ != 3) {
    throw error::ShapeError(
        "DecoderUnit::forward", "expected the shape of X to be in 3 dimensions",
        x_shape, error::make_message("received dimensions: ", rank_));
  }

  if (x_shape != y_shape) {
    throw error::ShapeError("DecoderUnit::forward",
                            "The shape of X and Y must be the same", x_shape,
                            error::make_message("received Y shape: ",
                                                error::format_shape(y_shape)));
  }

  if (x_shape[rank_ - 1] != d_model_) {
    throw error::ShapeError(
        "DecoderUnit::forward",
        "The last dimension of X or Y must be equal to d_model", x_shape,
        error::make_message("current d_model: ", d_model_));
  }

  ensure_buffers(X);

  source_ = context;

  ln_1.forward(X, *norm_1_output_);

  mha_self.forward(*norm_1_output_, *mha_self_output_, nullptr, mask);

  mha_self_output_->add(X);

  ln_2.forward(*mha_self_output_, *norm_2_output_);

  mha_cross.forward(*norm_2_output_, *mha_cross_output_, source_, mask);

  mha_cross_output_->add(*mha_self_output_);

  ln_3.forward(*mha_cross_output_, *norm_3_output_);
  ff_.forward(*norm_3_output_, Y);

  Y.add(*mha_cross_output_);
}

// TODO: Add more rigorous checking and error
// TODO: Need to add a general method/function to check the comparison shape and
// overlap check
void DecoderUnit::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                           bool accumulate) {

  require_training("DecoderUnit::backward");
  if (!source_ || source_->shape() != X.shape()) {
    error::StateError("DecoderUnit::backward: run forward() first");
  }
  ensure_gradients(X);

  ff_.backward(*norm_3_output_, dY, *d_norm_3_output_, false);

  ln_3.backward(*mha_cross_output_, *d_norm_3_output_, *d_mha_cross_output_,
                false);

  d_mha_cross_output_->add(dY);

  mha_cross.backward(*d_mha_cross_output_, *norm_2_output_, *source_,
                     *d_norm_2_output_, *dSource_, false);

  ln_2.backward(*mha_self_output_, *d_norm_2_output_, *d_mha_self_output_,
                false);

  d_norm_2_output_->add(*d_mha_cross_output_);

  mha_self.backward(*d_mha_self_output_, *norm_1_output_, *norm_1_output_,
                    *d_norm_query_, *d_norm_source_, false);

  d_norm_query_->add(*d_norm_source_);
  ln_1.backward(X, *d_norm_query_,
                *dX_); // FIXED: took dY as d_norm_source_ by mistake

  if (accumulate) {
    dX.add(*dX_);
    dX.add(*mha_self_output_);
  } else {
    dX = *dX_;
    dX.add(*mha_self_output_);
  }
}

DecoderLayer::DecoderLayer(size_t num_decoders, size_t d_model, size_t d_ff,
                           size_t heads, core::random::Generator *generator)
    : num_decoders_(num_decoders), d_model_(d_model), d_ff_(d_ff),
      heads_(heads) {
  for (size_t i = 0; i < num_decoders_; i++) {
    decoders_.insert(decoders_.begin() + i,
                     DecoderUnit(d_model_, d_ff_, heads_, generator));
    module_ptrs.insert(module_ptrs.begin() + i, &decoders_[i]);
  }
}

std::vector<Module *> DecoderLayer::modules() { return module_ptrs; }

std::vector<optional<Tensor> *> DecoderLayer::buffer_gradients_() {
  return {&grad_a_, &grad_b_};
};

void DecoderLayer::clear_local_cache() {
  Module::clear_local_cache();

  activations_.clear();
}

void DecoderLayer::ensure_buffers(const Tensor &X) {
  if (num_decoders_ == 0) {
    return;
  }

  activations_.reserve(num_decoders_ - 1);
  for (size_t i = 0; i < num_decoders_ - 1; i++) {
    if (i >= activations_.size()) {
      activations_.emplace_back(X.shape());
    } else if (activations_[i].shape() != X.shape()) {
      activations_[i] = Tensor(X.shape());
    }
  }

  // activations_.resize(num_decoders_ - 1);
}

void DecoderLayer::ensure_gradients(const Tensor &X) {
  utils::ensure(grad_a_, X.shape());
  utils::ensure(grad_b_, X.shape());
}

// The DecoderLayer forward pass
// passes the context to each decoder layer
void DecoderLayer::forward(const Tensor &X, Tensor &Y, const Tensor *context,
                           const Tensor *mask) {
  core::utils::require_nonoverlap(X, Y, "DecoderLayer::forward", "X", "Y");
  ensure_buffers(X);

  const Tensor *input = &X;
  for (std::size_t i = 0; i < num_decoders_; i++) {
    Tensor *output = &activations_[i];
    decoders_[i].forward(*input, *output, mask, context);

    input = output;
  }
}

void DecoderLayer::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                            bool accumulate) {

  require_training("DecoderLayer::backward");
  core_utils::require_nonoverlap(X, dX, "DecoderLayer::backward", "X", "dX");
  core_utils::require_nonoverlap(dY, dX, "DecoderLayer::backward", "dY", "dX");

  Tensor *current_grad = &*grad_a_;
  const Tensor &final_input =
      num_decoders_ == 1 ? X : activations_[num_decoders_ - 2];

  decoders_.back().backward(final_input, dY, *current_grad, false);
  for (std::size_t i = num_decoders_ - 1; i-- > 0;) {
    const Tensor &input = (i == 0) ? X : activations_[i - 1];

    Tensor *next_grad = (current_grad == &*grad_a_) ? &*grad_b_ : &*grad_a_;

    decoders_[i].backward(input, *current_grad, *next_grad, false);
  }

  if (accumulate)
    dX.add(*current_grad);
  else
    dX = *current_grad;
}

} // namespace nn::transformers
