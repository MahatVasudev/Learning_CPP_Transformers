#include "nn/module.h"
#include <core/errors.h>
#include <core/tensor.h>
#include <cstddef>
#include <nn/transformers/decoders.h>
#include <nn/utils.h>

using std::size_t;
namespace error = core::error;
namespace nn::transformers {

DecoderUnit::DecoderUnit(size_t d_model, size_t d_ff, size_t heads, size_t seed)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model, d_ff, seed + 1), ln_1(d_model), ln_2(d_model), ln_3(d_model),
      mha_self(d_model, heads, true), mha_cross(d_model, heads, false) {}

void DecoderUnit::set_training(bool state) {
  nn::Module::set_training(state);

  nn::Module::set_training_all(
      {&ff_, &ln_1, &ln_2, &ln_3, &mha_self, &mha_cross}, state);
}

void DecoderUnit::ensure_buffers(const Tensor &X) {
  utils::ensure(mha_self_output_, X.shape());
  utils::ensure(norm_1_output_, X.shape());
  utils::ensure(norm_2_output_, X.shape());
  utils::ensure(norm_3_output_, X.shape());
  utils::ensure(mha_cross_output_, X.shape());
}
void DecoderUnit::ensure_gradients(const Tensor &X) {
  utils::ensure(d_mha_self_output_, X.shape());
  utils::ensure(d_norm_query_, X.shape());
  utils::ensure(d_norm_source_, X.shape());
  utils::ensure(d_norm_2_output_, X.shape());
  utils::ensure(d_norm_3_output_, X.shape());
  utils::ensure(dX_, X.shape());
  utils::ensure(dSource_, X.shape());
  utils::ensure(d_mha_cross_output_, X.shape());
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
    // TODO: Throw error regarding running training first
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
  ln_1.backward(X, *d_norm_source_, *dX_);

  if (accumulate) {
    dX.add(*dX_);
    dX.add(*mha_self_output_);
  } else {
    dX = *dX_;
    dX.add(*mha_self_output_);
  }
}

} // namespace nn::transformers
