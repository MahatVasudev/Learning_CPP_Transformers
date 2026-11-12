#ifndef TRANFORMERS_TRANSFORMER_H

#define TRANFORMERS_TRANSFORMER_H

#include "core/random.h"
#include "nn/activations/softmax.h"
#include "nn/positional_encoding.h"
#include <cstddef>
#include <cstdint>
#include <nn/embedding.h>
#include <nn/module.h>
#include <nn/transformers/decoders.h>
#include <nn/transformers/encoders.h>
#include <optional>

using std::size_t;
using std::vector;

namespace nn::transformers {

class Transformer final : public nn::Module {
public:
  Transformer(size_t vocab, size_t d_model, size_t d_ff, size_t heads = 8,
              size_t n_encoders = 2, size_t n_decoders = 6,
              size_t max_len = 100000,
              std::optional<size_t> padding_idx = std::nullopt,
              core::random::Generator *generator = nullptr);

  // TODO: some how need to show Query and Result as Tensors but in int32
  //
  // i could make a simple type which would contain the int32_t type and a
  // conversion from this type to tensor, and no operation
  // what do i name it list? Input? maybe TokenizedTensor
  void forward(const vector<int32_t> &XQuery, const vector<int32_t> &XResult,
               Tensor &Y, const Tensor *mask = nullptr);

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

  vector<optional<Tensor> *> buffers_() override;
  vector<optional<Tensor> *> buffer_gradients_() override;
};

} // namespace nn::transformers

#endif // !TRANFORMERS_TRANSFORMER_H
