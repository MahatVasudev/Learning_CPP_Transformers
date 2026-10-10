#include "functional/gradcheck.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>

namespace functional {

GradcheckResult check_gradient(Tensor &x, const Tensor &analytic_grad,
                               const std::function<double()> &loss, double eps,
                               double tol) {
  assert(x.shape() == analytic_grad.shape());

  GradcheckResult grad_result = GradcheckResult();
  grad_result.passed = true;

  for (std::size_t n = 0; n < x.numel(); n++) {
    float original = x.data()[n];
    const float x_plus = static_cast<float>(original + eps);
    const float x_minus = static_cast<float>(original - eps);

    x.data()[n] = x_plus;
    const double loss_plus = loss();
    x.data()[n] = x_minus;
    const double loss_minus = loss();
    x.data()[n] = original;

    const double step =
        static_cast<float>(x_plus) - static_cast<float>(x_minus);
    double numeric = (loss_plus - loss_minus) / step;

    double analytic = analytic_grad.data()[n];

    double denom = std::max({std::abs(numeric), std::abs(analytic), 1e-8});
    double rel = std::abs(numeric - analytic) / denom;

    if (!std::isfinite(rel)) {

      grad_result.max_rel_error = rel;
      grad_result.analytic = analytic;
      grad_result.numeric = numeric;
      grad_result.worst_index = n;
      grad_result.passed = false;

      return grad_result;
    }

    if (rel > grad_result.max_rel_error) {
      grad_result.max_rel_error = rel;
      grad_result.analytic = analytic;
      grad_result.numeric = numeric;
      grad_result.worst_index = n;
    }
  }

  grad_result.passed = (grad_result.max_rel_error < tol);

  return grad_result;
}
} // namespace functional
