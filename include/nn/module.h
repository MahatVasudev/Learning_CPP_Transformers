#ifndef MODULE_FUNCTIONAL_H

#define MODULE_FUNCTIONAL_H
#include <cassert>
#include <core/tensor.h>
#include <optional>
#include <string>
#include <vector>

// #include "functional/tensor.h"

using core::Tensor;
namespace nn {

class Module {
public:
  virtual ~Module() = default;

  // parameters of all children modules
  virtual std::vector<Tensor *> parameters() {
    return concat_parameters(modules());
  };

  // gradients of all children modules
  virtual std::vector<Tensor *> gradients() {

    return concat_gradients(modules());
  };

  // this defines all of the children modules
  virtual std::vector<Module *> modules() { return {}; };

  // resets all of the gradients back to 0.0f
  void zero_grad() {
    for (Tensor *t : gradients()) {
      t->fill(0.0f);
    }
  }

  // sets training mode to true
  void train() { set_training(true); }

  // sets training mode to false
  void eval() { set_training(false); }

  // by default propagates all of the children modules to a given training state
  virtual void set_training(bool state) {
    training_ = state;
    for (Module *mod : modules()) {
      if (mod)
        mod->set_training(state);
    }
  }

  // returns the current training state
  bool training() const { return training_; }

  // automatically removes all of the cache present in the module and children
  // module
  void clear_cache();

protected:
  bool training_ = true;

  // goes through each modules and returns its parameters
  static std::vector<Tensor *>
  concat_parameters(const std::vector<Module *> &modules);
  // goes through each modules and returns its gradients
  static std::vector<Tensor *>
  concat_gradients(const std::vector<Module *> &modules);

  // goes through each modules and returns its parameters, plus any local
  // parameters
  std::vector<Tensor *>
  collect_parameters(const std::vector<Tensor *> parameters);

  // goes through each modules and returns its parameters, plus any local
  // gradients
  std::vector<Tensor *>
  collect_gradients(const std::vector<Tensor *> gradients);

  // helper to register each buffer for forward pass
  virtual std::vector<std::optional<Tensor> *> buffers_() { return {}; }

  // helper to register each buffer gradients for backward pass
  virtual std::vector<std::optional<Tensor> *> buffer_gradients_() {
    return {};
  }

  // by default removes all buffers_ and buffer_gradients_
  virtual void clear_local_cache() {
    for (std::optional<Tensor> *buff : buffers_()) {
      buff->reset();
    }

    for (std::optional<Tensor> *buff : buffer_gradients_()) {
      buff->reset();
    }
  };

  // helper to through error when the method can only be invoked during
  // training
  void require_training(std::string method_name,
                        std::string additional_message = "");
};

} // namespace nn

#endif // !MODULE_FUNCTIONAL_H
