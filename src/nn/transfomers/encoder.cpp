
#include "core/errors.h"
#include "core/tensor.h"
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

EncoderUnit::EncoderUnit(size_t d_model, size_t d_ff, size_t heads, size_t seed)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model_, d_ff_, seed + 1), mha_(d_model_, heads_, false),
      ln_1(d_model), ln_2(d_model) {}

std::vector<Tensor *> EncoderUnit::parameters() {
  return concat_parameters({&mha_, &ln_1, &ff_, &ln_2});
}

std::vector<Tensor *> EncoderUnit::gradients() {
  return concat_gradients({&mha_, &ln_1, &ff_, &ln_2});
}

void EncoderUnit::set_training(bool state) {
  nn::Module::set_training(state);
  nn::Module::set_training_all({&ff_, &mha_}, state);

  if (!state) {
    d_mha_output.reset();
    dX_.reset();
    dSource_.reset();
  }
}

// A single EncoderUnit forward
// uses PreNormalization i.e
// X + Sublayer(LayerNorm(X))
void EncoderUnit::forward(const Tensor &X, Tensor &Y) {
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
  mha_.forward(*norm_1_output_, *mha_output_);

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
  utils::ensure(mha_output_, X.shape());
  utils::ensure(norm_1_output_, X.shape());
  utils::ensure(norm_2_output_, X.shape());
}

void EncoderUnit::ensure_gradients(const Tensor &X) {
  utils::ensure(d_mha_output, X.shape());
  utils::ensure(dX_, X.shape());
  utils::ensure(dSource_, X.shape());
  utils::ensure(d_norm_2_output_, X.shape());
  utils::ensure(d_norm_1_output_, X.shape());
  // utils::ensure(dff_output_, X.shape());
}

} // namespace nn::transformers
