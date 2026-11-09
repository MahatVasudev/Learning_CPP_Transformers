#include "core/random.h"
#include "ops/matmul.h"
#include <cmath>
#include <core/errors.h>
#include <cstddef>
#include <limits>
#include <nn/module.h>
#include <nn/transformers/multiheadattention.h>
#include <nn/utils.h>
#include <vector>

using core::Tensor;
using nn::utils::ensure;

namespace error = core::error;

namespace nn::transformers {

MultiheadAttention::MultiheadAttention(std::size_t in_dims, std::size_t heads,
                                       bool is_causal,
                                       core::random::Generator *generator)
    : in_dims_(in_dims), heads_(heads), head_dims_(heads ? in_dims / heads : 0),
      is_causal_(is_causal), query_(in_dims, in_dims, false, generator),
      key_(in_dims, in_dims, false, generator),
      value_(in_dims, in_dims, false, generator),
      output_(in_dims, in_dims, true, generator) {

  if (heads_ == 0 || in_dims_ % heads_ != 0) {
    throw error::ShapeError("MultiheadAttention",
                            "in_dims should be divisible by heads", {in_dims},
                            error::make_message("got heads: ", heads));
  }
}

std::vector<Tensor *> MultiheadAttention::parameters() {
  return concat_parameters({&query_, &key_, &value_, &output_});
}

std::vector<Tensor *> MultiheadAttention::gradients() {
  return concat_gradients({&query_, &key_, &value_, &output_});
}

void MultiheadAttention::set_training(bool state) {
  Module::set_training(state);

  Module::set_training_all({&query_, &key_, &value_, &output_}, state);
  if (!state) {
    d_scores_.reset();
    d_probabilities_.reset();
    dQ_.reset();
    dK_.reset();
    dV_.reset();
  }
}

void MultiheadAttention::validate_buffers_(const Tensor &X,
                                           const Tensor &source) {
  // X: shape in_dim, d_model
  const std::vector<std::size_t> x_shape = X.shape();
  const std::vector<std::size_t> source_shape = source.shape();

  if (X.rank() != 3) {
    throw error::ShapeError(
        "MultiheadAttention",
        "X must have rank 3: (batch, sequence length, d_model)", x_shape);
  }

  if (source.rank() != 3) {

    throw error::ShapeError(
        "MultiheadAttention",
        "context must have rank 3: (batch, sequence length, d_model)", x_shape);
  }

  if (x_shape[0] != source_shape[0]) {

    throw error::ShapeError(
        "MultiheadAttention", "X and context must have the same batch length",
        source_shape,
        error::make_message("got X shape: ", error::format_shape(x_shape)));
  }

  if (x_shape[2] != in_dims_ || source_shape[2] != in_dims_) {

    throw error::ShapeError(
        "MultiheadAttention",
        "the last dimension (d_model) of X and context must equal in_dims",
        x_shape,
        error::make_message("got context shape: ",
                            error::format_shape(source_shape)));
  }
}

void MultiheadAttention::ensure_buffers_(const Tensor &X,
                                         const Tensor &source) {

  validate_buffers_(X, source);

  const std::vector<std::size_t> &x_shape = X.shape();
  const std::vector<std::size_t> &source_shape = source.shape();

  const std::vector<std::size_t> score_shape = {x_shape[0], heads_, x_shape[1],
                                                source_shape[1]};

  ensure(Q_proj_, x_shape);
  ensure(K_proj_, source_shape);
  ensure(V_proj_, source_shape);
  ensure(scores_, score_shape);
  ensure(probabilities_, score_shape);
  ensure(head_outputs_, x_shape);
  ensure(concatenated_, x_shape);
}

void MultiheadAttention::ensure_gradients_(const Tensor &X,
                                           const Tensor &source) {

  ensure_buffers_(X, source);

  const std::vector<std::size_t> &x_shape = X.shape();
  const std::vector<std::size_t> &source_shape = source.shape();

  const std::vector<std::size_t> score_shape = {x_shape[0], heads_, x_shape[1],
                                                source_shape[1]};

  ensure(dQ_, x_shape);
  ensure(dK_, source_shape);
  ensure(dV_, source_shape);
  ensure(d_scores_, score_shape);
  ensure(d_probabilities_, score_shape);
  // ensure(d_head_outputs_, x_shape);
}

void MultiheadAttention::mhattn_forward_check(const Tensor &X, const Tensor &Y,
                                              const Tensor *context) {
  if (X.rank() != 3 || X.shape().back() != in_dims_) {
    throw error::ShapeError(
        "MultiheadAttention::forward",
        error::make_message("X must have shape (B, Lq, ", in_dims_, ")"),
        X.shape());
  }

  if (Y.shape() != X.shape()) {
    throw error::ShapeError(
        "MultiheadAttention::forward", "Y must have the same shape as X",
        Y.shape(),
        error::make_message("X shape: ", error::format_shape(X.shape())));
  }

  const Tensor &source = context ? *context : X;
  if (source.rank() != 3 || source.shape()[0] != X.shape()[0] ||
      source.shape().back() != in_dims_) {
    throw error::ShapeError(
        "MultiheadAttention::forward",
        error::make_message("context must have shape (B, Lk, ", in_dims_,
                            ") with the same batch size as X"),
        source.shape());
  }
}

void MultiheadAttention::forward(const Tensor &X, Tensor &Y,
                                 const Tensor *context, const Tensor *mask) {

  mhattn_forward_check(X, Y, context);
  const Tensor &source = context ? *context : X;
  ensure_buffers_(X, source);

  const std::vector<std::size_t> x_shape = X.shape();
  const std::vector<std::size_t> source_shape = source.shape();

  if (x_shape[0] != source_shape[0]) {
    throw error::ShapeError(
        "MultiheadAttention::forward",
        "the batch of X (B) and Context (Bi) must be same", x_shape,
        error::make_message("got Context shape: ",
                            error::format_shape(source_shape)));
  }

  const std::size_t B = x_shape[0];
  const std::size_t S = x_shape[1];
  const std::size_t Bi = source_shape[0];
  const std::size_t Si = source_shape[1];

  const Tensor *causal_mask = nullptr;

  if (is_causal_) {
    if (!causal_mask_ || causal_mask_->shape()[2] != S ||
        causal_mask_->shape()[3] != Si) {

      causal_mask_ = create_causal_mask(S, Si);
    }

    causal_mask = &*causal_mask_;
  }

  query_.forward(X, *Q_proj_);
  key_.forward(source, *K_proj_);
  value_.forward(source, *V_proj_);

  Tensor K = K_proj_.value().view({Bi, Si, this->heads_, this->head_dims_});
  Tensor V = V_proj_.value().view({Bi, Si, this->heads_, this->head_dims_});
  Tensor Q = Q_proj_.value().view({B, S, this->heads_, this->head_dims_});

  Q = Q.permute({0, 2, 1, 3});
  K = K.permute({0, 2, 1, 3});
  V = V.permute({0, 2, 1, 3});

  ops::matmul(Q, K, *scores_, false, true, false);
  scores_->divide(std::sqrt(static_cast<float>(K.shape().back())));

  if (mask) {
    scores_->mask_fill(*mask, -std::numeric_limits<float>::infinity());
  }

  if (is_causal_) {
    scores_->mask_fill(*causal_mask, -std::numeric_limits<float>::infinity());
  }

  sm_.forward(*scores_, *probabilities_);
  ops::matmul(*probabilities_, V, *head_outputs_);

  Tensor ordered = head_outputs_->permute({0, 2, 1, 3}).contiguous();
  ordered.reshape({B, S, heads_ * head_dims_});

  *concatenated_ = std::move(ordered);

  output_.forward(*concatenated_, Y);
}

void MultiheadAttention::backward(const Tensor &dY, const Tensor &X,
                                  const Tensor &Source, Tensor &dX,
                                  Tensor &dSource, bool accumulate) {
  require_training("MultiheadAttention::backward");
  const std::vector<std::size_t> &dy_shape = dY.shape();
  ensure_gradients_(X, Source);
  const std::size_t B = X.shape()[0];
  const std::size_t Sq = X.shape()[1];
  const std::size_t Skv = Source.shape()[1];
  Tensor d_concatenated_(dy_shape);
  output_.backward(*concatenated_, dY, d_concatenated_, false);
  Tensor d_heads =
      d_concatenated_.view({dy_shape[0], dy_shape[1], heads_, head_dims_})
          .permute({0, 2, 1, 3});

  ops::matmul(*probabilities_, d_heads, *dV_, true, false, accumulate);
  Tensor V = V_proj_.value()
                 .view({B, Skv, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  Tensor K = K_proj_.value()
                 .view({B, Skv, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  Tensor Q = Q_proj_.value()
                 .view({B, Sq, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  ops::matmul(d_heads, V, *d_probabilities_, false, true, false);
  sm_.backward(*scores_, *probabilities_, *d_probabilities_, *d_scores_);

  d_scores_->divide(std::sqrt(static_cast<float>(head_dims_)));

  ops::matmul(*d_scores_, K, *dQ_, false, false, accumulate);
  ops::matmul(*d_scores_, Q, *dK_, true, false, accumulate);

  Tensor dQ_o = dQ_->permute({0, 2, 1, 3}).contiguous();
  Tensor dK_o = dK_->permute({0, 2, 1, 3}).contiguous();
  Tensor dV_o = dV_->permute({0, 2, 1, 3}).contiguous();

  dQ_o.reshape({B, Sq, in_dims_});
  dK_o.reshape({B, Skv, in_dims_});
  dV_o.reshape({B, Skv, in_dims_});

  query_.backward(X, dQ_o, dX, accumulate);
  key_.backward(Source, dK_o, dSource, accumulate);
  value_.backward(Source, dV_o, dSource, true);
}

Tensor MultiheadAttention::create_causal_mask(std::size_t query_length,
                                              std::size_t key_length,
                                              std::size_t query_offset) {

  if (query_length == 0 || key_length == 0 || query_offset >= key_length) {
    throw error::StateError(error::make_message(
        "MultiheadAttention::create_causal_mask: Got Invalid Configuration of "
        "query_length, key_length, query_offset; query_length, key_length > 0 "
        "and query_offset < key_length.\ncurrently got: query_length = ",
        query_length, " key_length = ", key_length,
        " query_offset = ", query_offset));
  }

  Tensor causal({1, 1, query_length, key_length});

  float *caus_data = causal.data();

  for (std::size_t i = 0; i < query_length; ++i) {
    const std::size_t row_offset = i * key_length;
    const std::size_t causal_limit = i + query_offset;

    for (std::size_t j = 0; j < key_length; j++) {
      caus_data[row_offset + j] = (j <= causal_limit) ? 1.0f : 0.0f;
    }
  }

  return causal;
}

} // namespace nn::transformers
