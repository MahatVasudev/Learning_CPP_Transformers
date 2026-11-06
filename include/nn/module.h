#ifndef MODULE_FUNCTIONAL_H

#include <cassert>
#include <core/tensor.h>
#include <string>
#include <vector>
#define MODULE_FUNCTIONAL_H

// #include "functional/tensor.h"

using core::Tensor;
namespace nn {

class Module {
public:
  virtual ~Module() = default;

  virtual std::vector<Tensor *> parameters() = 0;
  virtual std::vector<Tensor *> gradients() = 0;

  void zero_grad() {
    for (Tensor *t : gradients()) {
      t->fill(0.0f);
    }
  }

  void train() { set_training(true); }

  void eval() { set_training(false); }

  virtual void set_training(bool state) { training_ = state; }

  bool training() const { return training_; }

protected:
  bool training_ = true;
  static std::vector<Tensor *>
  concat_parameters(const std::vector<Module *> &modules);
  static std::vector<Tensor *>
  concat_gradients(const std::vector<Module *> &modules);
  static void set_training_all(const std::vector<Module *> &modules, bool mode);

  void require_training(std::string method_name,
                        std::string additional_message = "");
};

} // namespace nn

#endif // !MODULE_FUNCTIONAL_H
