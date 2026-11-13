#ifndef DECODER_GPT_TRANSFORMERS_NN_H

#define DECODER_GPT_TRANSFORMERS_NN_H

#include "nn/transformers/multiheadattention.h"
#include <core/random.h>
#include <nn/layer_norm.h>
#include <nn/module.h>
#include <nn/transformers/feedforward.h>
using core::random::Generator;
using std::vector;
namespace nn::transformers {

class SelfAttentionDecoder final : public Module {

public:
  SelfAttentionDecoder(size_t d_model, size_t d_ff, size_t heads,
                       bool causal_mask = true, Generator *generator = nullptr);

  vector<Module *> modules() override;

  void forward(const Tensor &X, Tensor &Y, const Tensor *mask);
  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

private:
  size_t d_model_, d_ff_, heads_;
  FeedForward ff_;
  LayerNorm ln_1, ln_2;
  MultiheadAttention mha_self;

  optional<Tensor> mha_self_output_, norm_1_output_, norm_2_output_;
  // FIXME: need to remove the norm 1 input X and source after solving for TN
  // 005
  optional<Tensor> d_mha_self_output_, d_norm_1_X_, d_norm_1_Source_,
      d_norm_2_output_;

  vector<optional<Tensor> *> buffers_() override;
  vector<optional<Tensor> *> buffer_gradients_() override;
};

class CrossAttentionDecoder final : public Module {

public:
  CrossAttentionDecoder(size_t d_model, size_t d_ff, size_t heads,
                        Generator *generator = nullptr);

  vector<Module *> modules() override;

  void forward(const Tensor &X, const Tensor &Source, Tensor &Y,
               const Tensor *mask);
  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

private:
  size_t d_model_, d_ff_, heads_;
  FeedForward ff_;
  LayerNorm ln_1, ln_2;
  MultiheadAttention mha_cross;

  std::optional<Tensor> mha_cross_output_, norm_1_output_;
  std::optional<Tensor> d_mha_self_output_, d_norm_1_output_, dX_;

  std::vector<optional<Tensor> *> buffers_() override;
  std::vector<optional<Tensor> *> buffer_gradients_() override;
};

} // namespace nn::transformers

#endif // !DECODER_GPT_TRANSFORMERS_NN_H
