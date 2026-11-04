#include "core/errors.h"
#include "core/tensor.h"
#include "nn/module.h"
#include <model/encoder.h>
#include <vector>

using core::Tensor;
namespace error = core::error;
using std::size_t;

// Encoder Unit
// Represents one encoder unit
// mha -> ff

namespace model {

EncoderUnit::EncoderUnit(size_t d_model, size_t d_ff, size_t heads, size_t seed)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model_, d_ff_, seed + 1), mha_(d_model_, heads_, false) {}

std::vector<Tensor *> EncoderUnit::parameters() {
  return concat_parameters({&ff_, &mha_});
}

std::vector<Tensor *> EncoderUnit::gradients() {
  return concat_gradients({&ff_, &mha_});
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

void EncoderUnit::forward(const Tensor &X, Tensor &Y) {
  const size_t &rank_ = X.rank();

  const std::vector<size_t> &x_shape = X.shape();
  const std::vector<size_t> &y_shape = Y.shape();
  if (x_shape != y_shape) {
    throw error::ShapeError("EncoderUnit::forward",
                            "The shape of X and Y must be the same", x_shape,
                            error::make_message("received Y shape: ",
                                                error::format_shape(y_shape)));
  }

  if (x_shape[rank_ - 1] != d_model_) {
    // TODO: throw x's last dimension must equal d_model_ previously provided
    throw error::ShapeError(
        "EncoderUnit::forward",
        "The last dimension of X or Y must be equal to d_model", x_shape,
        error::make_message("current d_model: ", d_model_));
  }

  ensure_buffers(X);

  mha_.forward(X, *mha_output_);

  ff_.forward(*mha_output_, Y);
}

void EncoderUnit::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                           bool accumulate) {
  require_training("EncoderUnit::backward");
  if (!mha_output_ || mha_output_->shape() != X.shape()) {
    throw error::StateError("EncoderUnit::backward: call forward() first");
  }
  ensure_gradients(X);
  ff_.backward(*mha_output_, dY, *d_mha_output, false);

  mha_.backward(*d_mha_output, X, X, *dX_, *dSource_, false);

  if (accumulate) {
    dX.add(*dX_);
    dX.add(*dSource_);
  } else {
    dX = *dX_;
    dX.add(*dSource_);
  }
}

void EncoderUnit::ensure_buffers(const Tensor &X) {
  if (!mha_output_ || mha_output_->shape() != X.shape()) {
    mha_output_.emplace(X.shape());
  }
}

void EncoderUnit::ensure_gradients(const Tensor &X) {
  if (!d_mha_output || d_mha_output->shape() != X.shape()) {
    d_mha_output.emplace(X.shape());
    dX_.emplace(X.shape());
    dSource_.emplace(X.shape());
  }
}

} // namespace model
