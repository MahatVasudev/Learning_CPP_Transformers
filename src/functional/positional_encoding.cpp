#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional/positional_encoding.h>
#include <functional/tensor.h>
#include <stdexcept>

double angle(std::size_t pos, std::size_t i, std::size_t d_model) {

  const double exponent =
      2.0 * static_cast<double>(i / 2) / static_cast<double>(d_model);

  return static_cast<double>(pos) / std::pow(10000.0, exponent);
}

namespace functional {

PositionalEncodingSinusoidal::PositionalEncodingSinusoidal(std::size_t max_len,
                                                           std::size_t d_model)
    : max_len_(max_len), d_model_(d_model),
      pos_enc_(Tensor({max_len, d_model})) {
  if (d_model_ % 2 != 0) {
    throw std::invalid_argument("d_model must be even");
  }
  build_table_();
}

void PositionalEncodingSinusoidal::add_to(Tensor &embedding) const {
  if (embedding.rank() < 2) {
    throw std::invalid_argument("Rank of embedding must be >= 2");
  }
  std::size_t rank = embedding.rank();
  std::size_t seq_len = embedding.shape()[rank - 2];
  if (seq_len > max_len_ || embedding.shape().back() != d_model_) {
    throw std::invalid_argument(
        "Last two dimension must match <=max_len, == d_model");
  }

  std::size_t rows = embedding.numel() / d_model_;

  float *emb = embedding.data();
  const float *pe = pos_enc_.data();

  for (std::size_t row = 0; row < rows; ++row) {
    float *e_row = emb + row * d_model_;
    const float *p_row = pe + (row % seq_len) * d_model_;

    for (std::size_t j = 0; j < d_model_; ++j) {
      e_row[j] += p_row[j];
    }
  }
}

void PositionalEncodingSinusoidal::build_table_() {

  float *t = pos_enc_.data();

  for (std::size_t pos = 0; pos < max_len_; ++pos) {
    float *row = t + pos * d_model_;

    for (std::size_t i = 0; i < d_model_; ++i) {
      row[i] = (i % 2 == 0) ? sine_value_(pos, i) : cosine_value_(pos, i);
    }
  }
}

float PositionalEncodingSinusoidal::sine_value_(const std::size_t pos,
                                                const std::size_t i) const {
  return static_cast<float>(std::sin(angle(pos, i, d_model_)));
}

float PositionalEncodingSinusoidal::cosine_value_(const std::size_t pos,
                                                  const std::size_t i) const {

  return static_cast<float>(std::cos(angle(pos, i, d_model_)));
}

} // namespace functional
  //
