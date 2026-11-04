#include <core/errors.h>
#include <core/utils.h>

namespace core::utils {

void require_contiguous(const Tensor &t, const std::string &operation,
                        const std::string &variable_name) {
  if (!t.is_contiguous()) {
    throw error::ShapeError(
        operation,
        error::make_message(variable_name.empty() ? "Tensor" : variable_name,
                            " needs to be contiguous"),
        t.shape(), "try running clone() first...");
  }
}

Tensor reshape(Tensor &tensor, std::vector<std::size_t> &shape) {
  Tensor t_diff = tensor;

  t_diff.reshape(shape);

  return t_diff;
}

} // namespace core::utils
