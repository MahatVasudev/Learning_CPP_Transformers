#include "core/random.h"
#include "core/utils.h"
#include "ops/matmul.h"
#include <algorithm>
#include <cmath>
#include <core/errors.h>
#include <cstddef>
#include <limits>
#include <nn/module.h>
#include <nn/transformers/multiheadattention.h>
#include <nn/utils.h>
#include <optional>
#include <vector>

using core::Tensor;
using nn::utils::ensure;
using std::optional;

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

std::vector<Module *> MultiheadAttention::modules() {
  return {&query_, &key_, &value_, &output_};
};

std::vector<optional<Tensor> *> MultiheadAttention::buffers_() {
  return {&Q_proj_,        &K_proj_,       &V_proj_,      &scores_,
          &probabilities_, &head_outputs_, &concatenated_};
}

std::vector<optional<Tensor> *> MultiheadAttention::buffer_gradients_() {
  return {&d_scores_, &d_probabilities_, &dQ_, &dK_, &dV_};
}

void MultiheadAttention::clear_local_cache() {
  Module::clear_local_cache();

  causal_mask_.reset();
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
  std::size_t query_offset = 0;

  const Tensor *causal_mask = nullptr;

  query_.forward(X, *Q_proj_);
  Tensor Q = Q_proj_->view({B, S, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  Tensor K(K_proj_->shape());
  Tensor V(V_proj_->shape());
  if (!cache_mode_) {
    key_.forward(source, *K_proj_);
    value_.forward(source, *V_proj_);
    K = K_proj_->view({Bi, Si, this->heads_, this->head_dims_})
            .permute({0, 2, 1, 3});
    V = V_proj_->view({Bi, Si, this->heads_, this->head_dims_})
            .permute({0, 2, 1, 3});

  } else {

    key_.forward(source, *K_proj_);
    value_.forward(source, *V_proj_);
    Tensor new_K = K_proj_->view({Bi, Si, this->heads_, this->head_dims_})
                       .permute({0, 2, 1, 3});
    Tensor new_V = V_proj_->view({Bi, Si, this->heads_, this->head_dims_})
                       .permute({0, 2, 1, 3});

    query_offset = kvCache->current_length();

    kvCache->append(new_K, new_V);
    K = kvCache->active_keys();
    V = kvCache->active_values();
  }

  if (is_causal_) {
    if (!causal_mask_ || causal_mask_->shape()[2] != S ||
        causal_mask_->shape()[3] != Si) {

      causal_mask_ = create_causal_mask(S, Si, query_offset);
    }

    causal_mask = &*causal_mask_;
  }

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
                                  const Tensor *Source, Tensor &dX,
                                  Tensor *dSource, bool accumulate) {
  require_training("MultiheadAttention::backward");
  const Tensor *source = (Source == nullptr) ? &X : Source;
  if ((dSource == nullptr) != (Source == nullptr))
    throw error::StateError("MultiheadAttention::backward: when providing "
                            "Source or dSource, you need to provide both");

  core::utils::require_nonoverlap(dY, dX, "MultiheadAttention", "dY", "dX");
  core::utils::require_nonoverlap(X, dX, "MultiheadAttention", "X", "dX");
  core::utils::require_nonoverlap(*source, dX, "MultiheadAttention", "Source",
                                  "dX");

  if (dSource != nullptr) {
    core::utils::require_nonoverlap(dY, *dSource, "MultiheadAttention", "dY",
                                    "dSource");
    core::utils::require_nonoverlap(X, *dSource, "MultiheadAttention", "X",
                                    "dSource");
    core::utils::require_nonoverlap(*source, *dSource, "MultiheadAttention",
                                    "Source", "dSource");

    core::utils::require_shape_match(Source->shape(), dSource->shape(),
                                     "MultiheadAttention", "Source", "dSource");
  }

  core::utils::require_shape_match(X.shape(), dX.shape(), "MultiheadAttention",
                                   "X", "dX");

  core::utils::require_shape_match(X.shape(), dY.shape(), "MultiheadAttention",
                                   "X", "dY");

  const std::vector<std::size_t> &dy_shape = dY.shape();
  ensure_gradients_(X, *source);
  const std::size_t B = X.shape()[0];
  const std::size_t Sq = X.shape()[1];
  const std::size_t Skv = source->shape()[1];
  Tensor d_concatenated_(dy_shape);
  output_.backward(*concatenated_, dY, d_concatenated_, false);
  Tensor d_heads =
      d_concatenated_.view({dy_shape[0], dy_shape[1], heads_, head_dims_})
          .permute({0, 2, 1, 3});

  ops::matmul(*probabilities_, d_heads, *dV_, true, false, false);
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

  ops::matmul(*d_scores_, K, *dQ_, false, false, false);
  ops::matmul(*d_scores_, Q, *dK_, true, false, false);

  Tensor dQ_o = dQ_->permute({0, 2, 1, 3}).contiguous();
  Tensor dK_o = dK_->permute({0, 2, 1, 3}).contiguous();
  Tensor dV_o = dV_->permute({0, 2, 1, 3}).contiguous();

  dQ_o.reshape({B, Sq, in_dims_});
  dK_o.reshape({B, Skv, in_dims_});
  dV_o.reshape({B, Skv, in_dims_});
  Tensor *dsource = (dSource == nullptr) ? &dX : dSource;
  query_.backward(X, dQ_o, dX, accumulate);
  key_.backward(*source, dK_o, *dsource,
                (dSource == nullptr) ? true : accumulate);
  value_.backward(*source, dV_o, *dsource, true);
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

void KVCache::append(const Tensor &new_keys, const Tensor &new_values) {
  const vector<size_t> &nk_shape = new_keys.shape();
  const vector<size_t> &nv_shape = new_values.shape();
  const vector<size_t> &ck_shape = keys.shape();
  const vector<size_t> &cv_shape = values.shape();

  if (nk_shape.size() != 4 || nv_shape.size() != 4 || ck_shape.size() != 4 ||
      cv_shape.size() != 4) {
    throw error::ShapeError(
        "KVCache::append",
        "expected 4 dimensions in keys and values, Keys shape", ck_shape,
        error::make_message(
            "Values shape: ", error::format_shape(cv_shape),
            "\nRecieved Keys shape: ", error::format_shape(nk_shape),
            " Recieved Value Shape: ", error::format_shape(nv_shape)));
  }

  if (nk_shape[0] != ck_shape[0] || nk_shape[1] != ck_shape[1] ||
      nk_shape[3] != ck_shape[3]) {
    throw error::ShapeError(
        "KVCache::append",
        "Recieved Keys Dimensions donot match, recieved keys", nk_shape,
        error::make_message(
            "current cache keys shape: ", error::format_shape(cv_shape),
            " dimensions to match (B, H, _, K)"));
  }

  if (nv_shape[0] != cv_shape[0] || nv_shape[1] != cv_shape[1] ||
      nv_shape[3] != cv_shape[3]) {
    throw error::ShapeError(
        "KVCache::append",
        "Recieved Values Dimensions donot match, recieved values", nk_shape,
        error::make_message(
            "current cache values shape: ", error::format_shape(cv_shape),
            " dimensions to match (B, H, _, V)"));
  }

  if (nv_shape[2] != nk_shape[2]) {
    throw error::ShapeError(
        "KVCache::append",
        "Sequence of the recieved Values and Keys donot match", nk_shape,
        error::make_message("Values shape: ", error::format_shape(cv_shape)));
  }

  size_t cpty = capacity();

  const size_t B = nk_shape[0];
  const size_t H = nk_shape[1];
  const size_t DK = nk_shape[3];
  const size_t DV = nv_shape[3];
  const size_t new_size = nk_shape[2];
  if (length > cpty || new_size > cpty - length) {
    if (is_static)
      throw error::AssertionError(
          "KVCache::append: Keys Values Cache Capacity Exceeded");
    else
      reserve(std::max(2 * capacity(), length + new_size));
  }

  for (size_t b = 0; b < B; b++) {
    for (size_t h = 0; h < H; h++) {
      for (size_t s = 0; s < new_size; s++) {

        size_t dst = ((b * H + h) * cpty + length + s);
        for (size_t dk = 0; dk < DK; dk++) {
          keys.data()[dst * DK + dk] = new_keys(b, h, s, dk);
        }

        for (size_t dv = 0; dv < DV; dv++) {
          values.data()[dst * DV + dv] = new_values(b, h, s, dv);
        }
      }
    }
  }

  this->length += new_size;
}

void KVCache::reserve(size_t new_capacity) {
  if (new_capacity <= capacity()) {
    return;
  }
  vector<size_t> nsk = keys.shape();
  nsk[2] = new_capacity;
  vector<size_t> nsv = values.shape();
  nsv[2] = new_capacity;

  Tensor new_keys(nsk);
  Tensor new_values(nsv);

  for (size_t b = 0; b < nsk[0]; b++) {

    for (size_t h = 0; h < nsk[1]; h++) {
      for (size_t s = 0; s < length; s++) {

        for (size_t k = 0; k < nsk[3]; k++)
          new_keys(b, h, s, k) = keys(b, h, s, k);

        for (size_t v = 0; v < nsv[3]; v++)
          new_values(b, h, s, v) = values(b, h, s, v);
      }
    }
  }

  keys = std::move(new_keys);
  values = std::move(new_values);
}

} // namespace nn::transformers
