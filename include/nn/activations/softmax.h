#ifndef SOFTMAX_ACTIVATIONS_NN_H

#include <core/tensor.h>

using core::Tensor;
#define SOFTMAX_ACTIVATIONS_NN_H

namespace nn::activations {

class Softmax {
public:
  explicit Softmax(int dim = -1) : dim_(dim) {}

  void forward(const Tensor &X, Tensor &Y) const;

  void backward(const Tensor &X, const Tensor &Y, const Tensor &dY, Tensor &dX,
                bool accumulate = false) const;

private:
  int dim_;
};

} // namespace nn::activations

#endif // !SOFTMAX_ACTIVATIONS_NN_H
