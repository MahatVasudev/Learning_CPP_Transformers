#include <core/tensor.h>
#include <core/utils.h>
#include <nn/activations/relu.h>

using core::Tensor;

using core::utils::require_contiguous;

namespace nn::activations {

void ReLU::forward(const Tensor &X, Tensor &Y) const {
  require_contiguous(X, "ReLU::forward", "X");
  require_contiguous(Y, "ReLU::forward", "Y");
  assert(Y.shape() == X.shape());
  assert(&X != &Y);
  const float *x = X.data();
  float *y = Y.data();
  std::size_t n = X.numel();
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

void ReLU::backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                    bool accumulate) const {
  require_contiguous(X, "ReLU::backward", "X");
  require_contiguous(dX, "ReLU::backward", "dX");
  require_contiguous(dY, "ReLU::backward", "dY");
  assert(X.shape() == dY.shape());
  assert(X.shape() == dX.shape());

  assert(&dX != &dY);
  assert(&dX != &X);

  const float *x = X.data();
  float *dx = dX.data();
  const float *dy = dY.data();

  std::size_t n = X.numel();
  for (std::size_t i = 0; i < n; ++i) {
    float g = x[i] > 0 ? dy[i] : 0;

    if (accumulate) {
      dx[i] += g;
    } else {
      dx[i] = g;
    }
  }
}
} // namespace nn::activations
