#ifndef MULTIHEADATTENTION_TRANSFORMERS_NN_H

#define MULTIHEADATTENTION_TRANSFORMERS_NN_H

#include "nn/activations/softmax.h"
#include <nn/linear.h>
#include <nn/module.h>
namespace nn::transformers {

class MultiheadAttention final : public Module {
public:
  MultiheadAttention(std::size_t in_dims, std::size_t heads,
                     bool is_causal = false);

  void forward(const Tensor &X, Tensor &Y, const Tensor *context = nullptr,
               const Tensor *mask = nullptr);

  void backward(const Tensor &dY, const Tensor &X, const Tensor &Source,
                Tensor &dX, Tensor &dSource, bool accumulate = false);

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;
  void set_training(bool state) override;

private:
  std::size_t in_dims_;
  std::size_t heads_;
  std::size_t head_dims_;
  bool is_causal_;
  nn::Linear query_, key_, value_, output_;
  nn::activations::Softmax sm_;

  std::optional<Tensor> Q_proj_, K_proj_, V_proj_, scores_, probabilities_,
      head_outputs_, concatenated_, d_scores_, causal_mask_, d_probabilities_,
      dQ_, dK_, dV_; // dQ_proj_, dK_proj_, dV_proj_, d_head_outputs_ removed

  void ensure_buffers_(const Tensor &X, const Tensor &source);

  void ensure_gradients_(const Tensor &X, const Tensor &source);

  void validate_buffers_(const Tensor &X, const Tensor &source);

  inline void mhattn_forward_check(const Tensor &X, const Tensor &Y,
                                   const Tensor *context);
  static Tensor create_causal_mask(std::size_t query_length,
                                   std::size_t key_length,
                                   std::size_t query_offset = 0);

}; // class MultiheadAttention
} // namespace nn::transformers

#endif // !MULTIHEADATTENTION_TRANSFORMERS_NN_H
