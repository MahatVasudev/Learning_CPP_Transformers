#include "core/random.h"
#ifndef EMBEDDING_NN_H

#include <core/tensor.h>
#include <cstddef>
#include <cstdint>
#include <nn/module.h>
#include <optional>

#define EMBEDDING_NN_H

namespace nn {

class Embedding final : public Module {
public:
  Embedding(std::size_t vocab_size, std::size_t d_model,
            std::optional<std::size_t> padding_idx = std::nullopt,
            core::random::Generator *generator = nullptr);
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  void forward(const std::int32_t *ids, std::size_t n, Tensor &Y) const;
  void backward(const std::int32_t *ids, std::size_t n, const Tensor &dY);

  void set_training(bool state) override;

  void ensure_gradient();

private:
  std::size_t vocab_size_;
  std::size_t d_model_;
  std::optional<std::size_t> padding_idx_;
  Tensor Emb_;
  std::optional<Tensor> dEmb_;
  void clear_local_cache() override;

  std::size_t checked_id_(std::int32_t id) const;
};
} // namespace nn

#endif // !EMBEDDING_NN_H
