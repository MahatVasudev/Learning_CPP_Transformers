#ifndef DECODER_TRANSFORMER_NN

#define DECODER_TRANSFORMER_NN

#include "core/random.h"
#include "nn/module.h"
#include "nn/transformers/feedforward.h"
#include <cstddef>
#include <nn/layer_norm.h>
#include <nn/transformers/multiheadattention.h>
#include <optional>
using std::size_t;

namespace nn::transformers {

class DecoderUnit final : public Module {

public:
  DecoderUnit(size_t d_model, size_t d_ff, size_t heads,
              core::random::Generator *gen = nullptr);

  void forward(const Tensor &X, Tensor &Y, const Tensor *mask = nullptr,
               const Tensor *context = nullptr);

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX, bool accomulate);

  std::vector<Module *> modules() override;

private:
  size_t d_model_, d_ff_, heads_;
  FeedForward ff_;
  LayerNorm ln_1, ln_2, ln_3;
  MultiheadAttention mha_self, mha_cross;
  const Tensor *source_ = nullptr;
  std::optional<Tensor> mha_self_output_, norm_1_output_, norm_2_output_,
      norm_3_output_, mha_cross_output_;
  std::optional<Tensor> d_mha_self_output_, d_norm_query_, d_norm_source_,
      d_norm_2_output_, d_norm_3_output_, dX_, dSource_, d_mha_cross_output_;

  std::vector<optional<Tensor> *> buffers_() override;
  std::vector<optional<Tensor> *> buffer_gradients_() override;
  void ensure_buffers(const Tensor &X);
  void clear_local_cache() override;
  void ensure_gradients(const Tensor &X);
};

class DecoderLayer final : public Module {

public:
  DecoderLayer(size_t num_decoders, size_t d_model, size_t d_ff, size_t heads,
               core::random::Generator *gen = nullptr);

  std::vector<Tensor> const &activations() const { return activations_; };

  void forward(const Tensor &X, Tensor &Y, const Tensor *context = nullptr,
               const Tensor *mask = nullptr);

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accomulate = false);

  std::vector<Module *> modules() override;

private:
  size_t num_decoders_;
  size_t d_model_;
  size_t d_ff_;
  size_t heads_;

  std::vector<Tensor> activations_;

  std::vector<DecoderUnit> decoders_;

  std::vector<Module *> module_ptrs;

  void clear_local_cache() override;
  std::vector<optional<Tensor> *> buffer_gradients_() override;
  std::optional<Tensor> grad_a_;
  std::optional<Tensor> grad_b_;
  void ensure_buffers(const Tensor &X);
  void ensure_gradients(const Tensor &X);
};

} // namespace nn::transformers

#endif // !DECODER_TRANSFORMER_NN
