#ifndef TRANFORMERS_TRANSFORMER_H

#define TRANFORMERS_TRANSFORMER_H

#include "core/random.h"
#include "nn/activations/softmax.h"
#include "nn/positional_encoding.h"
#include <cstddef>
#include <nn/embedding.h>
#include <nn/module.h>
#include <nn/transformers/decoders.h>
#include <nn/transformers/encoders.h>

using std::size_t;

namespace nn::transformers {

class Transformer final : public nn::Module {
public:
  Transformer(size_t vocab, size_t d_model, size_t d_ff, size_t n_encoders = 2,
              size_t n_decoders = 6,
              core::random::Generator *generator = nullptr);
  std::vector<Tensor *> parameters() override;

  std::vector<Tensor *> gradients() override;

  void forward(const Tensor &X, Tensor &Y, const Tensor *mask = nullptr);

  // TODO: define the parameters in backward
  void backward();

  void set_training(bool state) override;

private:
  size_t d_models_, vocab_, d_ff_, n_encoders_, n_decoders_;

  EncoderLayer encoders_;
  DecoderLayer decoders_;
  PositionalEncodingSinusoidal pos_enc;
  Embedding emb;
  activations::Softmax sfm;
  Linear out_linear;
};

} // namespace nn::transformers

#endif // !TRANFORMERS_TRANSFORMER_H
