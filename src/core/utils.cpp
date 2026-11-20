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

void require(bool value, const std::string &message) {
  if (!value) {
    throw error::AssertionError(message);
  }
}

void require_nonoverlap(const Tensor &t1, const Tensor &t2,
                        const std::string &operation,
                        const std::string &variable_name1,
                        const std::string &variable_name2,
                        const std::string &additional_message) {
  if (overlaps(t1.data(), t1.numel(), t2.data(), t2.numel())) {
    const std::string &t1_name =
        variable_name1.empty() ? "Tensor1" : variable_name1;

    const std::string &t2_name =
        variable_name2.empty() ? "Tensor2" : variable_name2;

    throw error::OverlapError(
        operation, error::make_message(t1_name, " overlaps with ", t2_name,
                                       "\n", additional_message));
  }
}

void require_shape_match(const vector<size_t> &shape1,
                         const vector<size_t> &shape2,
                         const std::string &operation,
                         const std::string &variable_name1,
                         const std::string &variable_name2,
                         const std::string &additional_message) {

  if (shape1 != shape2) {
    const std::string &t1_name =
        variable_name1.empty() ? "shape1" : "shape of " + variable_name1;

    const std::string &t2_name =
        variable_name2.empty() ? "shape2" : "shape of " + variable_name2;
    throw error::ShapeError(
        operation,
        error::make_message(variable_name1, "does not match; shape of ",
                            variable_name1),
        shape1,
        error::make_message(variable_name2, " ", error::format_shape(shape2),
                            "\n", additional_message));
  }
}

void require_broadcastable_shape(const vector<size_t> &shape1,
                                 const vector<size_t> &shape2,
                                 const std::string &operation,
                                 const std::string &variable_name1,
                                 const std::string &variable_name2,
                                 const std::string &additional_message) {

  if (Tensor::is_broadcast_possible(shape1, shape2)) {
    const std::string &t1_name =
        variable_name1.empty() ? "shape1" : "shape of " + variable_name1;

    const std::string &t2_name =
        variable_name2.empty() ? "shape2" : "shape of " + variable_name2;
    throw error::ShapeError(
        operation,
        error::make_message(variable_name1, "does cannot be broadcastable; ",
                            variable_name1),
        shape1,
        error::make_message(variable_name2, " ", error::format_shape(shape2),
                            "\n", additional_message));
  }
}

Tensor reshape(Tensor &tensor, std::vector<std::size_t> &shape) {
  Tensor t_diff = tensor;

  t_diff.reshape(shape);

  return t_diff;
}

bool overlaps(const float *p, std::size_t p_len, const float *q,
              const std::size_t q_len) {
  std::less<const float *> lt;
  return lt(p, q + q_len) && lt(q, p + p_len);
}

} // namespace core::utils
