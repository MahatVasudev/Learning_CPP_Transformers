#include "core/random.h"
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
                         size_t max_len, std::optional<size_t> padding_idx,
                         crandom::Generator *generator)
    : d_models_(d_model), vocab_(vocab), d_ff_(d_ff), n_encoders_(n_encoders),
      n_decoders_(n_decoders),
      encoders_(n_encoders_, d_model, d_ff, heads, generator),
      decoders_(n_decoders_, d_model, d_ff, heads, generator),
      pos_enc(max_len, d_model), emb(vocab, d_model, padding_idx, generator),
      out_linear(d_model, vocab, true, generator) {};

vector<Module *> Transformer::modules() {
  return {
      &encoders_,
      &decoders_,
      &emb,
      &out_linear,
  };
}

void Transformer::forward(const vector<int32_t> &XQuery,
                          const vector<int32_t> &XResult, Tensor &Y,
                          const Tensor *mask) {

  emb.forward(XQuery.data(), XQuery.size(), Y);
}

} // namespace nn::transformers
