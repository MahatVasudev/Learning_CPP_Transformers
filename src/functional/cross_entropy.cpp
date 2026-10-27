#include "functional/cross_entropy.h"
#include "functional/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional/functional.hpp>

namespace functional {
CrossEntropyResult cross_entropy(const Tensor &logits,
                                 const std::int32_t *targets,
                                 std::int32_t ignore_index) {
  require_contiguous(logits, "cross_entropy", "logits");
  assert(logits.rank() >= 1);
  assert(targets != nullptr);

  const std::size_t V = logits.shape().back();

  assert(V > 0);

  const std::size_t rows = logits.numel() / V;

  const float *z = logits.data();

  double total = 0.0;
  std::size_t count = 0;

  for (std::size_t r = 0; r < rows; ++r) {
    const std::int32_t t = targets[r];

    if (t == ignore_index)
      continue;

    assert(t >= 0 && static_cast<std::size_t>(t) < V);

    const float *row = z + r * V;

    float max_v = row[0];
    for (std::size_t j = 1; j < V; ++j) {
      max_v = std::max(max_v, row[j]);
    }

    double sum = 0.0;
    for (std::size_t j = 0; j < V; ++j) {
      sum += std::exp(static_cast<double>(row[j]) - max_v);
    }

    const double logsumexp = static_cast<double>(max_v) + std::log(sum);
    total += logsumexp - static_cast<double>(row[t]);
    ++count;
  }

  CrossEntropyResult result{0.0, count};

  if (count > 0) {
    result.loss = total / static_cast<double>(count);
  }

  return result;
}

void cross_entropy_backward(const Tensor &logits, const std::int32_t *targets,
                            Tensor &dlogits, const std::int32_t ignore_index,
                            bool accumulate) {
  require_contiguous(logits, "cross_entropy_backward", "logits");
  require_contiguous(dlogits, "cross_entropy_backward", "dlogits");
  assert(logits.rank() >= 1);
  assert(targets != nullptr);
  assert(dlogits.shape() == logits.shape());
  assert(&dlogits != &logits);

  const std::size_t V = logits.shape().back();
  assert(V > 0);
  const std::size_t rows = logits.numel() / V;

  std::size_t count = 0;

  for (std::size_t r = 0; r < rows; r++) {
    if (targets[r] != ignore_index) {
      ++count;
    }
  }

  if (count == 0) {
    if (!accumulate)
      dlogits.fill(0.0f);

    return;
  }

  const double inv_count = 1.0 / static_cast<double>(count);

  const float *z = logits.data();
  float *dz = dlogits.data();

  for (std::size_t r = 0; r < rows; ++r) {
    const std::int32_t t = targets[r];
    const float *row = z + r * V;
    float *g = dz + r * V;

    if (t == ignore_index) {
      if (!accumulate) {
        for (std::size_t j = 0; j < V; ++j) {
          g[j] = 0.0f;
        }
      }
      continue;
    }

    assert(t >= 0 && static_cast<std::size_t>(t) < V);

    float max_v = row[0];

    for (std::size_t j = 1; j < V; ++j) {
      max_v = std::max(max_v, row[j]);
    }

    double sum = 0.0;
    for (std::size_t j = 0; j < V; ++j) {
      sum += std::exp(static_cast<double>(row[j]) - max_v);
    }

    for (std::size_t j = 0; j < V; ++j) {
      const double p = std::exp(static_cast<double>(row[j]) - max_v) / sum;
      const double onehot = (static_cast<std::size_t>(t) == j) ? 1.0 : 0.0;
      const float val = static_cast<float>((p - onehot) * inv_count);

      if (accumulate) {
        g[j] += val;
      } else {
        g[j] = val;
      }
    }
  }
}
} // namespace functional
