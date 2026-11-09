#include "core/random.h"
#include <core/errors.h>
#include <core/tensor.h>
#include <core/utils.h>
#include <nn/embedding.h>
#include <nn/initialization.h>

using core::Tensor;
using core::utils::require_contiguous;
namespace error = core::error;

namespace nn {

Embedding::Embedding(std::size_t vocab_size, std::size_t d_model,
                     std::optional<std::size_t> padding_idx,
                     core::random::Generator *generator)
    : vocab_size_(vocab_size), d_model_(d_model), padding_idx_(padding_idx),
      Emb_({vocab_size, d_model}) {
  assert(vocab_size > 0 && d_model > 0);
  if (padding_idx_ && *padding_idx_ >= vocab_size_) {
    throw error::IndexError(
        "Embedding: padding_idx is out of bounds", *padding_idx_,
        error::make_message("padding_idx cannot be greater than ", vocab_size_,
                            " (vocab_size)"));
  }

  core::random::Generator &gen =
      (generator == nullptr) ? core::random::global_rng() : *generator;

  initialize::fill_random(Emb_, initialize::Initialize::normal(0.0f, 1.0f),
                          gen);

  if (padding_idx_) {
    float *row = Emb_.data() + *padding_idx_ * d_model_;
    std::fill(row, row + d_model_, 0.0f);
  }
}

std::size_t Embedding::checked_id_(std::int32_t id) const {
  std::size_t _id = static_cast<std::size_t>(id);
  if (id < 0 || _id >= vocab_size_) {
    throw error::IndexError(
        "Embedding id is out of range", _id,
        error::make_message("id cannot be less than 0 or greater than ",
                            vocab_size_, " (vocab_size)"));
  }

  return _id;
}

void Embedding::set_training(bool state) {
  Module::set_training(state);

  dEmb_.reset();
}

void Embedding::ensure_gradient() {
  if (!dEmb_) {

    dEmb_.emplace({vocab_size_, d_model_});
    dEmb_->fill(0.0f);
  }
}

void Embedding::forward(const std::int32_t *ids, std::size_t n,
                        Tensor &Y) const {
  require_contiguous(Y, "Embedding::forward", "Y");

  core::utils::require_nonoverlap(Y, Emb_, "Embedding::forward", "Y", "Emb_");
  assert(ids != nullptr);
  assert(Y.rank() >= 1 && Y.shape().back() == d_model_);
  assert(Y.numel() == n * d_model_);
  assert(&Y != &Emb_);

  const float *emb = Emb_.data();
  float *y = Y.data();

  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t id = checked_id_(ids[i]);
    const float *src = emb + id * d_model_;
    float *dst = y + i * d_model_;
    std::copy(src, src + d_model_, dst);
  }
}

void Embedding::backward(const std::int32_t *ids, std::size_t n,
                         const Tensor &dY) {

  if (!training_) {
    throw error::StateError("Embedding::backward: cannot run backward() on "
                            "eval(), run train() first");
  }
  require_contiguous(dY, "Embedding::forward", "Y");

  core::utils::require_nonoverlap(dY, Emb_, "Embedding::backward", "dY",
                                  "dEmb_");
  assert(ids != nullptr);
  assert(dY.rank() >= 1 && dY.shape().back() == d_model_);
  assert(dY.numel() == n * d_model_);

  ensure_gradient();
  const float *dy = dY.data();
  float *grad = dEmb_->data();

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t id = checked_id_(ids[i]);
    if (padding_idx_ && id == *padding_idx_)
      continue;
    const float *src = dy + i * d_model_;
    float *dst = grad + id * d_model_;
    for (std::size_t j = 0; j < d_model_; ++j) {
      dst[j] += src[j];
    }
  }
}

std::vector<Tensor *> Embedding::parameters() { return {&Emb_}; }

std::vector<Tensor *> Embedding::gradients() {
  ensure_gradient();
  return {&*dEmb_};
}

} // namespace nn
