#ifndef ENCODER_TRANSFORMERS_NN

#define ENCODER_TRANSFORMERS_NN
#include "core/tensor.h"
#include "nn/layer_norm.h"
#include "nn/module.h"
#include "nn/transformers/feedforward.h"
#include "nn/transformers/multiheadattention.h"
#include <cstddef>
#include <optional>

using core::Tensor;
using nn::Module;
using std::size_t;

namespace nn::transformers {

class EncoderUnit final : public Module {
public:
  EncoderUnit(size_t d_model, size_t d_ff, size_t heads_,
              core::random::Generator *gen = nullptr);

  void forward(const Tensor &X, Tensor &Y, const Tensor *mask = nullptr);

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
  LayerNorm ln_1, ln_2;

  std::optional<Tensor> mha_output_, norm_1_output_, norm_2_output_,
      d_mha_output_, dX_, dSource_, d_norm_1_output_, d_norm_2_output_;

  void ensure_buffers(const Tensor &X);
  void ensure_gradients(const Tensor &X);
};

class EncoderLayer final : public Module {

public:
  EncoderLayer(size_t num_layers, size_t d_model, size_t d_ff, size_t heads,
               core::random::Generator *gen = nullptr);
  std::vector<Tensor> const &activations() const { return activations_; };

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;
  void set_training(bool state) override;

  // NOTE: runs the entire num_encoders and in the end you can get input_buffers
  // as the entire results, or Y as our final result;
  void forward(const Tensor &X, Tensor &Y, const Tensor *mask);

  // NOTE: Returns one by one run of each num_encoders; useful for assigning
  // each encode output for each decoder input;
  void forward_gen(const Tensor &X, Tensor &Y, const Tensor *mask);

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

private:
  size_t num_encoders_;
  size_t d_model_;
  size_t d_ff_;
  size_t heads_;

  std::vector<EncoderUnit> encoders_;

  // contains size of num_encoders - 1
  std::vector<Tensor> activations_;
  // contains size of num_encoders - 1
  std::optional<Tensor> grad_a_;
  std::optional<Tensor> grad_b_;
  std::vector<Module *> module_ptrs_;

  // void _regenerate_module_ptrs();

  void ensure_buffers(const Tensor &X);
  void ensure_gradients(const Tensor &X);
};

} // namespace nn::transformers

#endif // !ENCODER_TRANSFORMERS_NN
