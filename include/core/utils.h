#ifndef CORE_TENSOR_UTILS

#include "string"
#include "tensor.h"
#define CORE_TENSOR_UTILS

using core::Tensor;
namespace core::utils {

bool overlaps(const float *p, std::size_t p_len, const float *q,
              const std::size_t q_len);
void require_contiguous(const Tensor &t, const std::string &operation = "",
                        const std::string &variable_name = "");
Tensor reshape(Tensor &tensor, std::vector<std::size_t> &shape);

void require_nonoverlap(const Tensor &t1, const Tensor &t2,
                        const std::string &operation,
                        const std::string &variable_name1 = "",
                        const std::string &variable_name2 = "",
                        const std::string &additional_message = "");

} // namespace core::utils
#endif // !CORE_TENSOR_UTILS
