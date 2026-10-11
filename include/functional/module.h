#ifndef MODULE_FUNCTIONAL_H

#define MODULE_FUNCTIONAL_H

#include "functional/tensor.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace functional {

class Module {
public:
  virtual ~Module() = default;

  virtual std::vector<Tensor *> parameters() = 0;
  virtual std::vector<Tensor *> gradients() = 0;

  void zero_grad() {
    for (Tensor *t : gradients()) {
      t->fill(0.0f);
    }
  };

  virtual void train() { training_ = true; }

  virtual void eval() { training_ = false; }

protected:
  bool training_;
  static std::vector<Tensor *>
  concat_parameters(const std::vector<Module *> modules);
  static std::vector<Tensor *>
  concat_gradients(const std::vector<Module *> modules);
};

class LayerNorm final : public Module {
public:
  LayerNorm(std::size_t normalized_shape, float epsilon = 1e-5f)
      : normalized_shape_(normalized_shape), epsilon_(epsilon),
        gamma_(Tensor({normalized_shape})), beta_(Tensor({normalized_shape})),
        gamma_grad_(Tensor({normalized_shape})),
        beta_grad_(Tensor({normalized_shape})) {
    gamma_.fill(1.0f);
    beta_.fill(0.0f);
    gamma_grad_.fill(0.0f);
    beta_grad_.fill(0.0f);
  }

  void forward(const Tensor &X, Tensor &Y);
  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

  const Tensor &gamma() const { return gamma_; }
  const Tensor &beta() const { return beta_; }
  const Tensor &gamma_grad() const { return gamma_grad_; }
  const Tensor &beta_grad() const { return beta_grad_; }

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

private:
  std::size_t normalized_shape_;
  float epsilon_;
  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_grad_;
  Tensor beta_grad_;

  float mean_calculate_(const float *x, std::size_t by) const;

  float variance_calculate_(const float *x, const float mean,
                            std::size_t by) const;
};

class Linear final : public Module {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true,
         unsigned seed = 0);

  void forward(const Tensor &X, Tensor &Y) const;

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

  std::vector<Tensor *> parameters() override;

  std::vector<Tensor *> gradients() override;

private:
  std::size_t in_, out_;
  bool bias_;
  Tensor W_, B_, dW_, dB_;
};

class Embedding final : public Module {
public:
  Embedding(std::size_t vocab_size, std::size_t d_model, unsigned seed = 0,
            std::optional<std::size_t> padding_idx = std::nullopt);
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  void forward(const std::int32_t *ids, std::size_t n, Tensor &Y) const;
  void backward(const std::int32_t *ids, std::size_t n, const Tensor &dY);

private:
  std::size_t vocab_size_;
  std::size_t d_model_;
  std::optional<std::size_t> padding_idx_;
  Tensor Emb_;
  Tensor dEmb_;

  std::size_t checked_id_(std::int32_t id) const;
};

} // namespace functional

#endif // !MODULE_FUNCTIONAL_H
