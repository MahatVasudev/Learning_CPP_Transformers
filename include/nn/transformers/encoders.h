#ifndef ENCODER_TRANSFORMERS_NN

#define ENCODER_TRANSFORMERS_NN
#include "core/tensor.h"
#include "nn/module.h"
#include "nn/transformers/feedforward.h"
#include "nn/transformers/multiheadattention.h"
#include <optional>

using core::Tensor;
using nn::Module;
using std::size_t;

namespace nn::transformers {

class EncoderUnit final : public Module {
  EncoderUnit(size_t d_model, size_t d_ff, size_t heads_, size_t seed);

  void forward(const Tensor &X, Tensor &Y);

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX, bool accumulate);

  std::vector<Tensor *> parameters() override;

  std::vector<Tensor *> gradients() override;

  void set_training(bool state) override;

private:
  size_t d_model_;
  size_t d_ff_;
  size_t heads_;
  FeedForward ff_;
  MultiheadAttention mha_;

  std::optional<Tensor> mha_output_, d_mha_output, dX_, dSource_;

  void ensure_buffers(const Tensor &X);
  void ensure_gradients(const Tensor &X);
};

class EncoderLayer final : public Module {};

} // namespace nn::transformers

#endif // !ENCODER_TRANSFORMERS_NN
