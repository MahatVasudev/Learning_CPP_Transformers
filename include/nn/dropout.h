#ifndef DROPOUT_NN_H

#define DROPOUT_NN_H
#include <core/random.h>
#include <nn/module.h>
#include <optional>

namespace crandom = core::random;

namespace nn {

class DropOut final : public Module {

public:
  // TODO: Need to optimize using 8-16 bit floating point for dropout

  DropOut(float p = 0.3, crandom::Generator *generator = nullptr);

  void forward(Tensor &X);

  void backward(const Tensor &dY, Tensor &dX, bool accumulate = false);

  float constexpr scale() const noexcept { return 1.0f / (1.0f - p_); }

private:
  float p_;
  crandom::Generator *rng_;
  std::optional<Tensor> mask_;
  vector<std::optional<Tensor> *> buffers_() override;

  void ensure_mask(const Tensor &X);
};
} // namespace nn

#endif // !DROPOUT_NN_H
