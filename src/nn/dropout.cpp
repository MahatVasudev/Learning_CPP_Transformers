#include "core/random.h"
#include "nn/utils.h"
#include <core/errors.h>
#include <nn/dropout.h>
#include <vector>

using std::vector;

namespace error = core::error;
namespace nn {

DropOut::DropOut(float p, crandom::Generator *generator)
    : p_(p), rng_(nullptr) {
  if (p > 1.0f || p < 0.0f) {
    throw error::StateError(error::make_message(
        "Dropout: p needs to be between 0.0f and 1.0f.\ncurrently got p: ", p));
  }
  rng_ = (generator == nullptr) ? &crandom::global_rng() : generator;
}

vector<std::optional<Tensor> *> DropOut::buffers_() { return {&mask_}; }

void DropOut::ensure_mask(const Tensor &X) { utils::ensure(mask_, X.shape()); }

void DropOut::forward(Tensor &X) {

  // IMP: check two things, whether we are in training mode or not
  // Secondly whether p=0.0, if either of them are true, just return empty early

  if (!training_ || p_ == 0.0f) {
    return;
  }

  ensure_mask(X);

  float *x_data = X.data();
  float *mask_data = mask_->data();

  Tensor::for_each(X.shape(), X.strides(), mask_->strides(),
                   [&](size_t input_offset, size_t output_offset, size_t) {
                     const bool keep = rng_->uniform() >= p_;
                     mask_data[output_offset] = keep ? 1.0f : 0.0f;
                     x_data[input_offset] *= mask_data[output_offset] * scale();
                   });
}

void DropOut::backward(const Tensor &dY, Tensor &dX, bool accumulate) {
  require_training("DropOut::backward");

  if (p_ == 0.0f) {
    if (accumulate) {
      dX.add(dY);
    } else {
      dX = dY;
    }
    return;
  }

  if (!mask_ || mask_->shape() != dX.shape()) {
    throw error::StateError("DropOut::backward: run forward() first");
  }

  float *x_data = dX.data();

  const float *y_data = dY.data();

  float *mask_data = mask_->data();

  Tensor::for_each(dX.shape(), dX.strides(), dY.strides(),
                   [&](size_t x_offset, size_t y_offset, size_t linear) {
                     const float grad =
                         y_data[y_offset] * mask_data[linear] * scale();

                     if (accumulate) {
                       x_data[x_offset] += grad;
                     } else {

                       x_data[x_offset] = grad;
                     }
                   });
}
} // namespace nn
