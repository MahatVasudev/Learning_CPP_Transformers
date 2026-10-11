#ifndef ACTIVATIONS_FUNC_H

#define ACTIVATIONS_FUNC_H

#include "functional/tensor.h"
namespace functional {

class Activation {
public:
  virtual ~Activation() = default;

  virtual void forward(const Tensor &X, Tensor &Y) const = 0;

  virtual void backward(const Tensor &X, const Tensor &Y, const Tensor &dY,
                        Tensor &dX, bool accumulate = false) const = 0;
};

class ReLU final : public Activation {
public:
  void forward(const Tensor &X, Tensor &Y) const override;

  void backward(const Tensor &X, [[maybe_unused]] const Tensor &Y,
                const Tensor &dY, Tensor &dX,
                bool accumulate = false) const override;
};

class Softmax final : public Activation {
public:
  explicit Softmax(int dim = -1) : dim_(dim) {}

  void forward(const Tensor &X, Tensor &Y) const override;

  void backward(const Tensor &X, const Tensor &Y, const Tensor &dY, Tensor &dX,
                bool accumulate = false) const override;

private:
  int dim_;
};

} // namespace functional

#endif // !ACTIVATIONS_FUNC_H
