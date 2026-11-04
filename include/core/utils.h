#ifndef CORE_TENSOR_UTILS

#define CORE_TENSOR_UTILS

#include "tensor.h"

using core::Tensor;
namespace core::utils {

void require_contiguous(const Tensor &t, const std::string &operation = "",
                        const std::string &variable_name = "");

} // namespace core::utils
#endif // !CORE_TENSOR_UTILS
