#include <nn/transformers/decoders_gpt.h>

namespace nn::transformers {

SelfAttentionDecoder::SelfAttentionDecoder(size_t d_model, size_t d_ff,
                                           size_t heads, bool causal_mask,
                                           Generator *generator)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model, d_ff, generator), ln_1(d_model), ln_2(d_model),
      mha_self(d_model, heads, causal_mask, generator) {}

vector<Module *> SelfAttentionDecoder::modules() {
  return {&ln_1, &mha_self, &ln_2, &ff_};
}

vector<optional<Tensor> *> SelfAttentionDecoder::buffers_() {
  return {&norm_1_output_, &mha_self_output_, &norm_2_output_};
}

vector<optional<Tensor> *> SelfAttentionDecoder::buffer_gradients_() {
  return {&d_norm_1_Source_, &d_norm_1_X_, &d_mha_self_output_,
          &d_norm_2_output_};
}

void SelfAttentionDecoder::forward(const Tensor &X, Tensor &Y,
                                   const Tensor *mask) {
  // TODO: TN 004

  ln_1.forward(X, *norm_1_output_);

  mha_self.forward(*norm_1_output_, *mha_self_output_, nullptr, mask);

  mha_self_output_->add(X);

  ln_2.forward(*mha_self_output_, *norm_2_output_);

  ff_.forward(*norm_2_output_, Y);

  Y.add(*mha_self_output_);
}

void SelfAttentionDecoder::backward(const Tensor &X, const Tensor &dY,
                                    Tensor &dX, bool accumulate) {

  // TODO: TN 004
  ff_.backward(*norm_2_output_, dY, *d_norm_2_output_, false);

  // X is already added
  ln_2.backward(*mha_self_output_, *d_norm_2_output_, *d_mha_self_output_,
                false);

  d_mha_self_output_->add(dY);

  // FIX: need to fix for mha taking source, TN 005
  mha_self.backward(*d_mha_self_output_, *norm_1_output_, *norm_1_output_,
                    *d_norm_1_X_, *d_norm_1_Source_, false);

  d_norm_1_X_->add(*d_norm_1_Source_);

  ln_1.backward(X, *d_norm_1_X_, dX, accumulate);

  dX.add(*d_mha_self_output_);
}
} // namespace nn::transformers
