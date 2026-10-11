#ifndef POSITIONAL_ENCODING_FUNCTIONAL_H

#define POSITIONAL_ENCODING_FUNCTIONAL_H

#include <cstddef>
#include <functional/tensor.h>

namespace functional {

class PositionalEncodingSinusoidal {
public:
  PositionalEncodingSinusoidal(std::size_t max_len, std::size_t d_model);
  Tensor *encoding() { return &pos_enc_; }
  const Tensor *encoding() const { return &pos_enc_; }

  void add_to(Tensor &embedding) const;

private:
  std::size_t max_len_;
  std::size_t d_model_;
  Tensor pos_enc_;

  void build_table_();
  float sine_value_(const std::size_t pos, const std::size_t i) const; // sine

  float cosine_value_(const std::size_t pos,
                      const std::size_t i) const; // cosine
};

} // namespace functional
#endif // !POSITIONAL_ENCODING_FUNCTIONAL_H
