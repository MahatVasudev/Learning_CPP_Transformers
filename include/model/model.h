#ifndef MODEL_MODEL_H

#define MODEL_MODEL_H
#include <functional/activations.h>
#include <functional/module.h>

namespace model {

class FeedForward final : public functional::Module {

public:
  FeedForward(std::size_t d_model, std::size_t d_ff, unsigned seed = 0);

  void forward(const functional::Tensor &X, functional::Tensor &Y);
  void backward(const functional::Tensor &X, const functional::Tensor &Y,
                functional::Tensor &dX, bool accumulate = false);
  std::vector<functional::Tensor *> parameters() override;

  std::vector<functional::Tensor *> gradients() override;

private:
  std::size_t d_model_;
  std::size_t d_ff_;
  functional::Linear l1_;
  functional::Linear l2_;
  functional::ReLU relu_;

  std::optional<functional::Tensor> h_pre_, h_act_, dh_pre_, dh_act_;
  std::vector<std::size_t> hidden_shape_(const functional::Tensor &X) const;
  void ensure_buffers_(const functional::Tensor &X);
};
} // namespace model
#endif // !MODEL_MODEL_H
