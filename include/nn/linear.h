#ifndef LINEAR_NN

#define LINEAR_NN

#include <core/random.h>
#include <core/tensor.h>
#include <nn/module.h>
#include <optional>

using core::Tensor;

namespace nn {

class Linear final : public Module {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias = true,
         core::random::Generator *generator = nullptr);

  void forward(const Tensor &X, Tensor &Y) const;

  void backward(const Tensor &X, const Tensor &dY, Tensor &dX,
                bool accumulate = false);

  std::vector<Tensor *> parameters() override;

  std::vector<Tensor *> gradients() override;

  void set_training(bool state) override;

  void clear_cache() override;

private:
  std::size_t in_, out_;
  bool bias_;
  Tensor W_, B_;
  std::optional<Tensor> dW_, dB_;

  void ensure_gradients_();
};

} // namespace nn

#endif // !LINEAR_NN
