#ifndef LAYER_NORM_NN_H

#include <core/tensor.h>
#include <nn/module.h>
#include <optional>

#define LAYER_NORM_NN_H

namespace nn {

class LayerNorm final : public Module {
public:
  LayerNorm(std::size_t normalized_shape, float epsilon = 1e-5f)
      : normalized_shape_(normalized_shape), epsilon_(epsilon),
        gamma_(Tensor({normalized_shape})), beta_(Tensor({normalized_shape})) {
    gamma_.fill(1.0f);
    beta_.fill(0.0f);
  }

  void forward(const Tensor &X, Tensor &Y) const;
  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

  const Tensor &gamma() const { return gamma_; }
  const Tensor &beta() const { return beta_; }
  const Tensor &gamma_grad() const { return *gamma_grad_; }
  const Tensor &beta_grad() const { return *beta_grad_; }

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  void set_training(bool state) override;

private:
  std::size_t normalized_shape_;
  float epsilon_;
  Tensor gamma_, beta_;
  std::optional<Tensor> gamma_grad_, beta_grad_;
  void clear_local_cache() override;
  float mean_calculate_(const float *x, std::size_t by) const;

  float variance_calculate_(const float *x, const float mean,
                            std::size_t by) const;

  void ensure_gradients_();
};
} // namespace nn

#endif // !LAYER_NORM_NN_H

#define LAYER_NORM_NN_H
