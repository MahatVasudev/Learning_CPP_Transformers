#ifndef TRANFORMERS_TRANSFORMER_H

#define TRANFORMERS_TRANSFORMER_H

#include "core/random.h"
#include "nn/activations/softmax.h"
#include "nn/dropout.h"
#include "nn/positional_encoding.h"
#include <core/tokenized_tensor.h>
#include <cstddef>
#include <nn/embedding.h>
#include <nn/module.h>
#include <nn/transformers/decoders.h>
#include <nn/transformers/encoders.h>
#include <optional>

using core::TokenizedTensor;
using std::size_t;
using std::vector;

namespace nn::transformers {

class Transformer final : public nn::Module {
public:
  // TODO: Need to optimize using 8-16 bit floating point for dropout

  Transformer(size_t vocab, size_t d_model, size_t d_ff, size_t heads = 8,
              size_t n_encoders = 2, size_t n_decoders = 6, float dropout = 0.1,
              size_t max_len = 100000,
              std::optional<size_t> padding_idx = std::nullopt,
              core::random::Generator *generator = nullptr);

  // IMPLEMENTED: using TokenizedTensor type which holds shape strides and int
  // data can be converted to Tensor when needed

  // Transformer forward pass, made specifically for training. does not save
  // Encoder Output in cache
  void forward_train(const TokenizedTensor &XQuery,
                     const TokenizedTensor &XResult, Tensor &Y,
                     const Tensor *query_mask = nullptr,
                     const Tensor *output_mask = nullptr);
  void encode(const TokenizedTensor &XQuery, Tensor &Y,
              const Tensor *query_mask = nullptr);
  void decode(const TokenizedTensor &XResult, Tensor &Y,
              const Tensor *context = nullptr,
              const Tensor *output_mask = nullptr);
  // TODO: define the parameters in backward
  void backward();

  vector<Module *> modules() override;

private:
  size_t d_models_, vocab_, d_ff_, n_encoders_, n_decoders_;

  EncoderLayer encoders_;
  DecoderLayer decoders_;
  PositionalEncodingSinusoidal pos_enc;
  Embedding emb;
  activations::Softmax sfm;
  Linear out_linear;
  DropOut drop_out_enc;
  DropOut drop_out_dec;

  optional<Tensor> query_embedding_, query_encoded_, result_embedding_;

  vector<optional<Tensor> *> buffers_() override;
  vector<optional<Tensor> *> buffer_gradients_() override;
};

} // namespace nn::transformers

#endif // !TRANFORMERS_TRANSFORMER_H
