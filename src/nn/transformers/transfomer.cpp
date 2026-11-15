#include "core/random.h"
#include "nn/dropout.h"
#include "nn/positional_encoding.h"
#include <core/errors.h>
#include <core/tensor.h>
#include <cstddef>
#include <nn/transformers/transformer.h>
#include <nn/utils.h>
#include <vector>

using core::Tensor;
using std::size_t;
using std::vector;
namespace crandom = core::random;
namespace error = core::error;

namespace nn::transformers {

Transformer::Transformer(size_t vocab, size_t d_model, size_t d_ff,
                         size_t heads, size_t n_encoders, size_t n_decoders,
                         float dropout, size_t max_len,
                         std::optional<size_t> padding_idx,
                         crandom::Generator *generator)
    : d_models_(d_model), vocab_(vocab), d_ff_(d_ff), n_encoders_(n_encoders),
      n_decoders_(n_decoders),
      encoders_(n_encoders_, d_model, d_ff, heads, generator),
      decoders_(n_decoders_, d_model, d_ff, heads, generator),
      pos_enc(max_len, d_model), emb(vocab, d_model, padding_idx, generator),
      out_linear(d_model, vocab, true, generator),
      drop_out_enc(dropout, generator), drop_out_dec(dropout, generator) {};

vector<Module *> Transformer::modules() {
  return {
      &encoders_, &decoders_, &emb, &out_linear, &drop_out_enc, &drop_out_dec,
  };
}

void Transformer::encode(const TokenizedTensor &XQuery, Tensor &Y,
                         const Tensor *query_mask) {

  emb.forward(XQuery.data(), XQuery.numel(), *query_embedding_);

  pos_enc.add_to(*query_embedding_);

  drop_out_enc.forward(*query_embedding_);

  encoders_.forward(*query_embedding_, Y, query_mask);
}

void Transformer::decode(const TokenizedTensor &XResult, Tensor &Y,
                         const Tensor *context, const Tensor *output_mask) {

  emb.forward(XResult.data(), XResult.numel(), *result_embedding_);

  pos_enc.add_to(*result_embedding_);

  drop_out_dec.forward(*result_embedding_);
  decoders_.forward(*result_embedding_, Y, context, output_mask);
}

void Transformer::forward_train(const TokenizedTensor &XQuery,
                                const TokenizedTensor &XResult, Tensor &Y,
                                const Tensor *input_mask,
                                const Tensor *output_mask) {
  // TODO: TN 004 ! add check for 3 dimensional data in Tokenized-Tensor, the
  // batch size and the vocab size of Query and Result must be same might not be
  // needed as Embedding is inside of the Transformer class

  // TODO: Need to check whether emb will correctly assign the data to
  // query_embedding buffer, w.r.t the shape
  encode(XQuery, *query_encoded_, input_mask);

  decode(XResult, Y, &*query_encoded_, output_mask);
}

} // namespace nn::transformers
