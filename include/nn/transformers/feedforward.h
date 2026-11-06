#ifndef TRANSFORMERS_NN_FEEDFORWARD

#define TRANSFORMERS_NN_FEEDFORWARD

#include <nn/activations/relu.h>
#include <nn/linear.h>
#include <nn/module.h>
#include <optional>
namespace nn::transformers {

class FeedForward final : public Module {

public:
  FeedForward(std::size_t d_model, std::size_t d_ff, unsigned seed = 0);

  void forward(const Tensor &X, Tensor &Y);
  void backward(const Tensor &X, const Tensor &Y, Tensor &dX,
                bool accumulate = false);
  std::vector<Tensor *> parameters() override;

  std::vector<Tensor *> gradients() override;

  void set_training(bool state) override;

private:
  std::size_t d_model_;
  std::size_t d_ff_;
  nn::Linear l1_;
  nn::Linear l2_;
  nn::activations::ReLU relu_;

  std::optional<Tensor> h_pre_, h_act_, dh_pre_, dh_act_;
  std::vector<std::size_t> hidden_shape_(const Tensor &X) const;
  void ensure_buffers_(const Tensor &X);

  void ensure_gradients_(const Tensor &X);
}; // class FeedForward

} // namespace nn::transformers

#endif // !TRANSFORMERS_NN_FEEDFORWARD
