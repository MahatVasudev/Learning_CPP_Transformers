#ifndef OPS_UTILS

#define OPS_UTILS

#include <core/tensor.h>
#include <cstddef>
#include <vector>

namespace ops::utils {

// batch_offset, calculating batch offset of tensor Tensor
std::size_t batch_offset(std::size_t flat_batch,
                         const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &strides);

// batch_offset, calculating offset on a broad-cast Tensor
std::size_t batch_offset(const std::vector<std::size_t> &batch_index,
                         const std::size_t &batch_rank,
                         const std::size_t &input_batch_rank,
                         const std::vector<std::size_t> &input_shape,
                         const std::vector<std::size_t> &input_stride);

std::vector<std::size_t> matmul_shape(const std::vector<std::size_t> shape_a,
                                      const std::vector<std::size_t> shape_b,
                                      bool transpose_a = false,
                                      bool transpose_b = false);

std::vector<std::size_t> matmul_shape(const core::Tensor &A,
                                      const core::Tensor &B,
                                      bool transpose_a = false,
                                      bool tranpose_b = false);
} // namespace ops::utils

#endif // !OPS_UTILS
