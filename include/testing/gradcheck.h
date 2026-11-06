#ifndef GRADCHECK_FUNCTIONAL_H

#include "core/tensor.h"
#include <functional>
#define GRADCHECK_FUNCTIONAL_H

namespace testing {

struct GradcheckResult {
  double max_rel_error;
  std::size_t worst_index;
  double analytic;
  double numeric;
  bool passed;
};

GradcheckResult check_gradient(core::Tensor &x,
                               const core::Tensor &analytic_grad,
                               const std::function<double()> &loss,
                               double eps = 1e-3, double tol = 1e-2);
} // namespace testing

#endif // !GRADCHECK_FUNCTIONAL_H
