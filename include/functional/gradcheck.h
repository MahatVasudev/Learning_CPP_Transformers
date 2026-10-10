#ifndef GRADCHECK_FUNCTIONAL_H

#define GRADCHECK_FUNCTIONAL_H

#include "functional/tensor.h"

#include <functional>

namespace functional {

struct GradcheckResult {
  double max_rel_error;
  std::size_t worst_index;
  double analytic;
  double numeric;
  bool passed;
};

GradcheckResult check_gradient(Tensor &x, const Tensor &analytic_grad,
                               const std::function<double()> &loss,
                               double eps = 1e-3, double tol = 1e-2);
} // namespace functional

#endif // !GRADCHECK_FUNCTIONAL_H
