#ifndef CROSS_FUNCTIONAL_H

#define CROSS_FUNCTIONAL_H

#include "core/tensor.h"
#include <cstddef>
#include <cstdint>

using core::Tensor;
namespace ops {

struct CrossEntropyResult {
  double loss;
  std::size_t count;
};

CrossEntropyResult cross_entropy(const Tensor &logits,
                                 const std::int32_t *targets,
                                 std::int32_t ignore_index = -100);

void cross_entropy_backward(const Tensor &logits, const std::int32_t *targets,
                            Tensor &dlogits, std::int32_t ignore_index = -100,
                            bool accumulate = false);

} // namespace ops

#endif // !CROSS_FUNCTIONAL_H
