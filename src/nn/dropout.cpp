#include "core/random.h"
#include <nn/dropout.h>
#include <vector>

using std::vector;

namespace nn {

DropOut::DropOut(float p, crandom::Generator *generator)
    : p_(p), rng_(nullptr) {
  if (p > 1.0f || p < 0.0f) {
    // TODO: Throw error;
  }
  rng_ = (generator == nullptr) ? &crandom::global_rng() : generator;
}

vector<std::optional<Tensor> *> DropOut::buffers_() { return {&mask_}; }

void DropOut::forward(Tensor &X) {

  // IMP: check two things, whether we are in training mode or not
  // Secondly whether p=0.0, if either of them are true, just return empty early

  if (!training_ || p_ == 0.0f) {
    return;
  }
  float *x_data = X.data();
  float *mask_data = mask_->data();

  const float scale = 1.0f / (1.0f - p_);
  Tensor::for_each(X.shape(), X.strides(), mask_->strides(),
                   [&](size_t input_offset, size_t output_offset, size_t) {
                     const bool keep = rng_->uniform() >= p_;
                     mask_data[output_offset] = keep ? 1.0f : 0.0f;
                     x_data[input_offset] *= mask_data[output_offset] * scale;
                   });
}

void DropOut::backward(const Tensor &dY, Tensor &dX, bool accumulate) {
  // TODO: TN 004
  // check whether mask was made, and mask has the same shape as dX, else call
  // forward first check whether it is in training or not, and call error if it
  // is not

  if (p_ == 0.0f) {
    if (accumulate) {
      dX.add(dY);
    } else {
      dX = dY;
    }
    return;
  }

  float *x_data = dX.data();

  const float *y_data = dY.data();

  float *mask_data = mask_->data();

  const float scale = 1.0f / (1.0f - p_);
  Tensor::for_each(dX.shape(), dX.strides(), dY.strides(),
                   [&](size_t x_offset, size_t y_offset, size_t linear) {
                     const float grad =
                         y_data[y_offset] * mask_data[linear] * scale;

                     if (accumulate) {
                       x_data[x_offset] += grad;
                     } else {

                       x_data[x_offset] = grad;
                     }
                   });
}
} // namespace nn
