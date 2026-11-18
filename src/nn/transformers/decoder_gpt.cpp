#include "core/utils.h"
#include <core/errors.h>
#include <nn/transformers/decoders_gpt.h>
#include <nn/utils.h>

namespace error = core::error;
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
  return {&d_norm_1_X_, &d_mha_self_output_, &d_norm_2_output_};
}

void SelfAttentionDecoder::ensure_buffers(const Tensor &X) {
  for (optional<Tensor> *buf : buffers_()) {
    utils::ensure(*buf, X.shape());
  }
}

void SelfAttentionDecoder::ensure_buffer_gradients(const Tensor &X) {
  for (optional<Tensor> *buf : buffer_gradients_()) {
    utils::ensure(*buf, X.shape());
  }
}

void SelfAttentionDecoder::forward(const Tensor &X, Tensor &Y,
                                   const Tensor *mask) {
  core::utils::require_nonoverlap(X, Y, "SelfAttentionDecoder::forward", "X",
                                  "Y");

  if (X.shape() != Y.shape() || X.rank() != 3) {
    throw error::ShapeError(
        "SelfAttentionDecoder::forward",
        "X and Y Tensor must be same, and rank must be == 3;", X.shape(),
        error::make_message("Got Y tensor shape = ",
                            error::format_shape(Y.shape())));
  }
  ensure_buffers(X);

  ln_1.forward(X, *norm_1_output_);

  mha_self.forward(*norm_1_output_, *mha_self_output_, nullptr, mask);

  mha_self_output_->add(X);

  ln_2.forward(*mha_self_output_, *norm_2_output_);

  ff_.forward(*norm_2_output_, Y);

  Y.add(*mha_self_output_);
}

void SelfAttentionDecoder::backward(const Tensor &X, const Tensor &dY,
                                    Tensor &dX, bool accumulate) {

  require_training("SelfAttentionDecoder::backward");

  core::utils::require_nonoverlap(X, dX, "SelfAttentionDecoder::backward", "X",
                                  "dX");
  core::utils::require_nonoverlap(dY, dX, "SelfAttentionDecoder::backward",
                                  "dY", "dX");
  if (!norm_2_output_ || norm_2_output_->shape() != X.shape()) {
    throw error::StateError(
        "SelfAttentionDecoder::backward(): run forward() first");
  }

  ensure_buffer_gradients(X);
  ff_.backward(*norm_2_output_, dY, *d_norm_2_output_, false);

  // X is already added
  ln_2.backward(*mha_self_output_, *d_norm_2_output_, *d_mha_self_output_,
                false);

  d_mha_self_output_->add(dY);

  mha_self.backward(*d_mha_self_output_, *norm_1_output_, nullptr, *d_norm_1_X_,
                    nullptr, false);

  d_norm_1_X_->add(*d_norm_1_Source_);

  ln_1.backward(X, *d_norm_1_X_, dX, accumulate);

  dX.add(*d_mha_self_output_);
}

// Implementing CrossAttentionDecoder

CrossAttentionDecoder::CrossAttentionDecoder(size_t d_model, size_t d_ff,
                                             size_t heads, Generator *generator)
    : d_model_(d_model), d_ff_(d_ff), heads_(heads),
      ff_(d_model, d_ff, generator), ln_1(d_model), ln_2(d_model),
      mha_cross(d_model, heads, false, generator) {};

vector<Module *> CrossAttentionDecoder::modules() {
  return {&ln_1, &mha_cross, &ln_2, &ff_};
}

vector<optional<Tensor> *> CrossAttentionDecoder::buffers_() {
  return {&mha_cross_output_, &norm_1_output_, &norm_2_output_};
}

vector<optional<Tensor> *> CrossAttentionDecoder::buffer_gradients_() {
  return {&d_mha_cross_output_, &d_norm_1_output_, &d_norm_2_output_,
          &d_norm_source_output};
}

void CrossAttentionDecoder::forward(const Tensor &X, const Tensor &Source,
                                    Tensor &Y, const Tensor *mask) {
  // TODO: TN 004

  core::utils::require_nonoverlap(X, Y, "CrossAttentionDecoder::forward", "X",
                                  "Y");
  core::utils::require_nonoverlap(Source, Y, "CrossAttentionDecoder::forward",
                                  "Source", "Y");
  if (X.shape() != Y.shape() || X.rank() != 3) {
    throw error::ShapeError(
        "CrossAttentionDecoder::forward",
        "X and Y Tensor must be same, and rank must be == 3; (batch, sequence, "
        "d_model)",
        X.shape(),
        error::make_message("Got Y tensor shape = ",
                            error::format_shape(Y.shape())));
  }

  if (Source.rank() != 3) {
    throw error::ShapeError(
        "CrossAttentionDecoder::forward",
        "Source needs to have 3 dimensional data (batch, sequence, d_model)",
        Source.shape());
  }
  *source = Source;
  ln_1.forward(X, *norm_1_output_);

  mha_cross.forward(*norm_1_output_, *mha_cross_output_, &Source, mask);

  mha_cross_output_->add(X);

  ln_2.forward(*mha_cross_output_, *norm_2_output_);

  ff_.forward(*norm_2_output_, Y);

  Y.add(*mha_cross_output_);
}

void CrossAttentionDecoder::backward(const Tensor &X, const Tensor &dY,
                                     Tensor &dX, bool accumulate) {

  // TODO: TN 004

  require_training("CrossAttentionDecoder::backward");

  core::utils::require_nonoverlap(X, dX, "CrossAttentionDecoder::backward", "X",
                                  "dX");
  core::utils::require_nonoverlap(dY, dX, "CrossAttentionDecoder::backward",
                                  "dY", "dX");
  if (source == nullptr) {
    throw error::StateError(
        "SelfAttentionDecoder::backward(): run forward() first");
  }
  ff_.backward(X, dY, *norm_2_output_, false);

  ln_2.backward(*mha_cross_output_, *d_norm_2_output_, *d_mha_cross_output_,
                false);

  d_mha_cross_output_->add(dY);

  mha_cross.backward(*d_mha_cross_output_, *norm_2_output_, *source,
                     *d_norm_1_output_, *d_norm_source_output, false);

  d_norm_1_output_->add(*d_norm_source_output);

  ln_1.backward(X, *d_norm_1_output_, dX, accumulate);

  dX.add(*d_mha_cross_output_);
}

} // namespace nn::transformers
