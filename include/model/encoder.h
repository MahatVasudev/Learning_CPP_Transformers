
#ifndef ENCODER_MODEL_H

#define ENCODER_MODEL_H

#include "core/tensor.h"
#include "nn/model.h"
#include "nn/module.h"

using core::Tensor;
using nn::Module;
using std::size_t;

namespace model {

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

  nn::FeedForward ff_;
  nn::MultiheadAttention mha_;

  std::optional<Tensor> mha_output_, d_mha_output, dX_, dSource_;

  void ensure_buffers(const Tensor &X);
  void ensure_gradients(const Tensor &X);
};

class EncoderLayer final : public Module {};

} // namespace model

#endif // !ENCODER_MODEL_H
