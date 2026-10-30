#ifndef MODULE_FUNCTIONAL_H

#define MODULE_FUNCTIONAL_H

#include "errors/errors.h"
#include "functional/tensor.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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
  }

  void train() { set_training(true); }

  void eval() { set_training(false); }

  virtual void set_training(bool state) { training_ = state; }

  bool training() const { return training_; }

protected:
  bool training_ = true;
  static std::vector<Tensor *>
  concat_parameters(const std::vector<Module *> &modules);
  static std::vector<Tensor *>
  concat_gradients(const std::vector<Module *> &modules);
  static void set_training_all(const std::vector<Module *> &modules, bool mode);

  void require_training(std::string method_name,
                        std::string additional_message = "");
};

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

  float mean_calculate_(const float *x, std::size_t by) const;

  float variance_calculate_(const float *x, const float mean,
                            std::size_t by) const;

  void ensure_gradients_();
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

  void set_training(bool state) override;

private:
  std::size_t in_, out_;
  bool bias_;
  Tensor W_, B_;
  std::optional<Tensor> dW_, dB_;

  void ensure_gradients_();
};

class Embedding final : public Module {
public:
  Embedding(std::size_t vocab_size, std::size_t d_model, unsigned seed = 0,
            std::optional<std::size_t> padding_idx = std::nullopt);
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  void forward(const std::int32_t *ids, std::size_t n, Tensor &Y) const;
  void backward(const std::int32_t *ids, std::size_t n, const Tensor &dY);

  void set_training(bool state) override;

  void ensure_gradient();

private:
  std::size_t vocab_size_;
  std::size_t d_model_;
  std::optional<std::size_t> padding_idx_;
  Tensor Emb_;
  std::optional<Tensor> dEmb_;

  std::size_t checked_id_(std::int32_t id) const;
};

} // namespace functional

#endif // !MODULE_FUNCTIONAL_H
