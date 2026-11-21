#ifndef MULTIHEADATTENTION_TRANSFORMERS_NN_H

#define MULTIHEADATTENTION_TRANSFORMERS_NN_H

#include "core/random.h"
#include "nn/activations/softmax.h"
#include <nn/linear.h>
#include <nn/module.h>
#include <optional>

using std::optional;
namespace nn::transformers {

// KVCache:
// stores the cache for keys and values in MultiHeadAttention
// KVCache owns the keys and values, rather than referencing it
struct KVCache {
  Tensor keys;
  Tensor values;

  KVCache(size_t batch_size, size_t heads, size_t max_length, size_t key_dim,
          size_t value_dim, bool is_static = true)
      : keys({batch_size, heads, max_length, key_dim}),
        values({batch_size, heads, max_length, value_dim}),
        is_static(is_static) {};

  size_t capacity() const { return keys.shape()[2]; }
  void append(const Tensor &new_keys, const Tensor &new_values);
  void reset() { length = 0; };
  size_t current_length() const { return length; }

  Tensor active_keys() const { return keys.narrow(2, 0, length); }
  Tensor active_values() const { return values.narrow(2, 0, length); }

  void reserve(size_t new_capacity);

private:
  std::size_t length = 0;
  const bool is_static;
};

// TODO: TN 005
class MultiheadAttention final : public Module {
public:
  MultiheadAttention(std::size_t in_dims, std::size_t heads,
                     bool is_causal = false,
                     core::random::Generator *generator = nullptr);

  void forward(const Tensor &X, Tensor &Y, const Tensor *context = nullptr,
               const Tensor *mask = nullptr);

  void backward(const Tensor &dY, const Tensor &X, const Tensor *Source,
                Tensor &dX, Tensor *dSource, bool accumulate = false);

  void enable_cache(bool is_static = true) {
    cache_mode_ = true;
    cache_static_ = is_static;
  };

  void reset_cache();
  std::vector<Module *> modules() override;

private:
  size_t in_dims_;
  size_t heads_;
  size_t head_dims_;
  bool is_causal_;
  nn::Linear query_, key_, value_, output_;
  nn::activations::Softmax sm_;
  bool cache_mode_ = false;
  bool cache_ = true;

  std::optional<KVCache> kvCache;
  std::optional<Tensor> Q_proj_, K_proj_, V_proj_, scores_, probabilities_,
      head_outputs_, concatenated_, d_scores_, causal_mask_, d_probabilities_,
      dQ_, dK_, dV_;

  void ensure_buffers_(const Tensor &X, const Tensor &source);

  void ensure_gradients_(const Tensor &X, const Tensor &source);

  std::vector<optional<Tensor> *> buffers_() override;

  std::vector<optional<Tensor> *> buffer_gradients_() override;

  void clear_local_cache() override;

  void validate_buffers_(const Tensor &X, const Tensor &source);

  inline void mhattn_forward_check(const Tensor &X, const Tensor &Y,
                                   const Tensor *context);
  static Tensor create_causal_mask(std::size_t query_length,
                                   std::size_t key_length,
                                   std::size_t query_offset = 0);

}; // class MultiheadAttention
} // namespace nn::transformers

#endif // !MULTIHEADATTENTION_TRANSFORMERS_NN_H
