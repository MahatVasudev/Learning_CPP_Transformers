#ifndef RELU_ACTIVATION_NN_H

#include <core/tensor.h>

using core::Tensor;

#define RELU_ACTIVATION_NN_H

namespace nn::activations {
class ReLU {
public:
  void forward(const Tensor &X, Tensor &Y) const;

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false) const;
};

} // namespace nn::activations

#endif // !RELU_ACTIVATION_NN_H
