#include "functional/activations.h"
#include "functional/ops.h"
#include <cmath>
#include <cstddef>
#include <errors/errors.h>
#include <functional/module.h>
#include <model/model.h>
#include <vector>

using functional::Tensor;

namespace model {

void ensure(std::optional<functional::Tensor> &buffer,
            const std::vector<std::size_t> &shape) {

  if (buffer && buffer->shape() == shape) {
    return;
  }

  buffer.emplace(shape);
}

// FeedForward Implementation -------------------------------

FeedForward::FeedForward(std::size_t d_model, std::size_t d_ff, unsigned seed)
    : d_model_(d_model), d_ff_(d_ff),
      l1_(functional::Linear(d_model, d_ff, true, seed)),
      l2_(functional::Linear(d_ff, d_model, true, seed + 1)), relu_() {}

void FeedForward::set_training(bool state) {
  functional::Module::set_training(state);

  functional::Module::set_training_all({&l1_, &l2_}, state);

  if (!state) {
    dh_pre_.reset();
    dh_act_.reset();
  }
}

void FeedForward::ensure_gradients_(const functional::Tensor &X) {
  const std::vector<std::size_t> shape = this->hidden_shape_(X);
  if (dh_pre_ && dh_pre_->shape() == shape)
    return; // steady state: no allocation
  dh_pre_.emplace(shape);
  dh_act_.emplace(shape);
}

std::vector<std::size_t>
FeedForward::hidden_shape_(const functional::Tensor &X) const {
  if (X.rank() < 1 || X.shape().back() != d_model_)
    throw error::ShapeError(
        "FeedForward",
        error::make_message("last dimension of X must be equal to (d_model) ",
                            d_model_),
        X.shape());
  std::vector<std::size_t> shape = X.shape();
  shape.back() = d_ff_;
  return shape;
}

void FeedForward::ensure_buffers_(const functional::Tensor &X) {
  const std::vector<std::size_t> shape = this->hidden_shape_(X);
  if (h_pre_ && h_pre_->shape() == shape)
    return; // steady state: no allocation
  h_pre_.emplace(shape);
  h_act_.emplace(shape);
}

void FeedForward::forward(const functional::Tensor &X, functional::Tensor &Y) {
  if (Y.shape() != X.shape())
    throw error::ShapeError("FeedForward::forward",
                            "the shape of Y must be equal to X", Y.shape(),
                            error::make_message("the shape of X is ",
                                                error::format_shape(X.shape()))

    );
  ensure_buffers_(X);
  l1_.forward(X, *h_pre_);
  // std::cout << "pased l1";
  relu_.forward(*h_pre_, *h_act_);
  // std::cout << "pased l2";
  l2_.forward(*h_act_, Y);
  // std::cout << "pased l3";
}

void FeedForward::backward(const functional::Tensor &X,
                           const functional::Tensor &dY, functional::Tensor &dX,
                           bool accumulate) {
  require_training("FeedForward::backward");
  if (!h_pre_ || h_pre_->shape() != hidden_shape_(X))
    throw error::StateError("FeedForward::backward: call forward() first");
  if (dY.shape() != X.shape() || dX.shape() != X.shape())
    throw error::ShapeError(
        "FeedForward::backward", "X's shape must match with dY and dX",
        X.shape(),
        error::make_message("dX's shape: ", error::format_shape(dX.shape()),
                            "dY's shape: ", error::format_shape(dY.shape())));
  ensure_gradients_(X);

  l2_.backward(*h_act_, dY, *dh_act_, false);
  relu_.backward(*h_pre_, *h_act_, *dh_act_, *dh_pre_, false);
  l1_.backward(X, *dh_pre_, dX, accumulate);
}

std::vector<Tensor *> FeedForward::parameters() {
  return this->concat_parameters({&l1_, &l2_});
}

std::vector<Tensor *> FeedForward::gradients() {
  return this->concat_gradients({&l1_, &l2_});
}

// MultiheadAttention Implementation -------------------------------

MultiheadAttention::MultiheadAttention(std::size_t in_dims, std::size_t heads,
                                       bool is_causal)
    : in_dims_(in_dims), heads_(heads), head_dims_(heads ? in_dims / heads : 0),
      is_causal_(is_causal), query_(in_dims, in_dims, false, 1),
      key_(in_dims, in_dims, false, 2), value_(in_dims, in_dims, false, 3),
      output_(in_dims, in_dims, true, 4) {

  if (heads_ == 0 || in_dims_ % heads_ != 0) {
    throw error::ShapeError("MultiheadAttention",
                            "in_dims should be divisible by heads", {in_dims},
                            error::make_message("got heads: ", heads));
  }
}

std::vector<functional::Tensor *> MultiheadAttention::parameters() {
  return concat_parameters({&query_, &key_, &value_, &output_});
}

std::vector<functional::Tensor *> MultiheadAttention::gradients() {
  return concat_gradients({&query_, &key_, &value_, &output_});
}

void MultiheadAttention::set_training(bool state) {
  functional::Module::set_training(state);

  functional::Module::set_training_all({&query_, &key_, &value_, &output_},
                                       state);
  if (!state) {
    d_scores_.reset();
    d_probabilities_.reset();
    // d_head_outputs_.reset();
    // dQ_proj_.reset();
    // dK_proj_.reset();
    // dV_proj_.reset();
  }
}

void MultiheadAttention::validate_buffers_(const functional::Tensor &X,
                                           const functional::Tensor &source) {
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

void MultiheadAttention::ensure_buffers_(const functional::Tensor &X,
                                         const functional::Tensor &source) {

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

void MultiheadAttention::ensure_gradients_(const functional::Tensor &X,
                                           const functional::Tensor &source) {

  ensure_buffers_(X, source);

  const std::vector<std::size_t> &x_shape = X.shape();
  const std::vector<std::size_t> &source_shape = source.shape();

  const std::vector<std::size_t> score_shape = {x_shape[0], heads_, x_shape[1],
                                                source_shape[1]};

  // ensure(dQ_proj_, x_shape);
  // ensure(dK_proj_, source_shape);
  // ensure(dV_proj_, source_shape);
  ensure(d_scores_, score_shape);
  ensure(d_probabilities_, score_shape);
  // ensure(d_head_outputs_, x_shape);
}

void MultiheadAttention::mhattn_forward_check(
    const functional::Tensor &X, const functional::Tensor &Y,
    const functional::Tensor *context) {
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

  const functional::Tensor &source = context ? *context : X;
  if (source.rank() != 3 || source.shape()[0] != X.shape()[0] ||
      source.shape().back() != in_dims_) {
    throw error::ShapeError(
        "MultiheadAttention::forward",
        error::make_message("context must have shape (B, Lk, ", in_dims_,
                            ") with the same batch size as X"),
        source.shape());
  }
}

void MultiheadAttention::forward(const functional::Tensor &X,
                                 functional::Tensor &Y,
                                 const functional::Tensor *context,
                                 const functional::Tensor *mask) {

  mhattn_forward_check(X, Y, context);
  const functional::Tensor &source = context ? *context : X;
  ensure_buffers_(X, source);

  const std::vector<std::size_t> x_shape = X.shape();
  const std::vector<std::size_t> source_shape = source.shape();
  const std::size_t B = x_shape[0];
  const std::size_t S = x_shape[1];
  const std::size_t Bi = source_shape[0];
  const std::size_t Si = source_shape[1];

  query_.forward(X, *Q_proj_);
  key_.forward(source, *K_proj_);
  value_.forward(source, *V_proj_);

  Tensor K = K_proj_.value().view({Bi, Si, this->heads_, this->head_dims_});
  Tensor V = V_proj_.value().view({Bi, Si, this->heads_, this->head_dims_});
  Tensor Q = Q_proj_.value().view({B, S, this->heads_, this->head_dims_});

  Q = Q.permute({0, 2, 1, 3});
  K = K.permute({0, 2, 1, 3});
  V = V.permute({0, 2, 1, 3});

  functional::matmul(Q, K, *scores_, false, true, false);
  scores_->divide(std::sqrt(static_cast<float>(K.shape().back())));

  sm_.forward(*scores_, *probabilities_);
  functional::matmul(*probabilities_, V, *head_outputs_);

  Tensor heads = head_outputs_->permute({0, 2, 1, 3});
  Tensor ordered = heads.clone();
  ordered.reshape({B, S, heads_ * head_dims_});

  *concatenated_ = std::move(ordered);

  output_.forward(*concatenated_, Y);
}

void MultiheadAttention::backward(const Tensor &dY, const Tensor &X,
                                  const Tensor &Source, Tensor &dX,
                                  Tensor &dSource, Tensor &dQ, Tensor &dK,
                                  Tensor &dV, bool accumulate) {
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

  functional::matmul(*probabilities_, d_heads, dV, true, false, accumulate);
  Tensor V = V_proj_.value()
                 .view({B, Skv, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  Tensor K = K_proj_.value()
                 .view({B, Skv, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  Tensor Q = Q_proj_.value()
                 .view({B, Sq, this->heads_, this->head_dims_})
                 .permute({0, 2, 1, 3});

  functional::matmul(d_heads, V, *d_probabilities_, false, true, false);
  sm_.backward(*scores_, *probabilities_, *d_probabilities_, *d_scores_);

  d_scores_->divide(std::sqrt(static_cast<float>(head_dims_)));

  functional::matmul(*d_scores_, K, dQ, false, false, accumulate);
  functional::matmul(*d_scores_, Q, dK, true, false, accumulate);

  Tensor dQ_o = dQ.permute({0, 2, 1, 3}).clone();
  Tensor dK_o = dK.permute({0, 2, 1, 3}).clone();
  Tensor dV_o = dV.permute({0, 2, 1, 3}).clone();

  dQ_o.reshape({B, Sq, in_dims_});
  dK_o.reshape({B, Skv, in_dims_});
  dV_o.reshape({B, Skv, in_dims_});

  query_.backward(X, dQ_o, dX, accumulate);
  key_.backward(Source, dK_o, dSource, accumulate);
  value_.backward(Source, dV_o, dSource, true);
}

} // namespace model
