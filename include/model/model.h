#ifndef MODEL_MODEL_H

#define MODEL_MODEL_H
#include <functional/activations.h>
#include <functional/module.h>

namespace model {

inline void ensure(std::optional<functional::Tensor> &buffer,
                   const std::vector<std::size_t> &shape);

class FeedForward final : public functional::Module {

public:
  FeedForward(std::size_t d_model, std::size_t d_ff, unsigned seed = 0);

  void forward(const functional::Tensor &X, functional::Tensor &Y);
  void backward(const functional::Tensor &X, const functional::Tensor &Y,
                functional::Tensor &dX, bool accumulate = false);
  std::vector<functional::Tensor *> parameters() override;

  std::vector<functional::Tensor *> gradients() override;

  void set_training(bool state) override;

private:
  std::size_t d_model_;
  std::size_t d_ff_;
  functional::Linear l1_;
  functional::Linear l2_;
  functional::ReLU relu_;

  std::optional<functional::Tensor> h_pre_, h_act_, dh_pre_, dh_act_;
  std::vector<std::size_t> hidden_shape_(const functional::Tensor &X) const;
  void ensure_buffers_(const functional::Tensor &X);

  void ensure_gradients_(const functional::Tensor &X);
}; // class FeedForward

class MultiheadAttention final : public functional::Module {
public:
  MultiheadAttention(std::size_t in_dims, std::size_t heads,
                     bool is_causal = false);

  void forward(const functional::Tensor &X, functional::Tensor &Y,
               const functional::Tensor *context = nullptr,
               const functional::Tensor *mask = nullptr);

  void backward(const functional::Tensor &dY, const functional::Tensor &X,
                const functional::Tensor &Source, functional::Tensor &dX,
                functional::Tensor &dSource, functional::Tensor &dQ,
                functional::Tensor &dK, functional::Tensor &dV,
                bool accumulate = false);

  std::vector<functional::Tensor *> parameters() override;
  std::vector<functional::Tensor *> gradients() override;
  void set_training(bool state) override;

private:
  std::size_t in_dims_;
  std::size_t heads_;
  std::size_t head_dims_;
  bool is_causal_;
  functional::Linear query_, key_, value_, output_;
  functional::Softmax sm_;

  std::optional<functional::Tensor> Q_proj_, K_proj_, V_proj_, scores_,
      probabilities_, head_outputs_, concatenated_, d_scores_,
      d_probabilities_; // dQ_proj_, dK_proj_, dV_proj_, d_head_outputs_ removed

  void ensure_buffers_(const functional::Tensor &X,
                       const functional::Tensor &source);

  void ensure_gradients_(const functional::Tensor &X,
                         const functional::Tensor &source);

  void validate_buffers_(const functional::Tensor &X,
                         const functional::Tensor &source);

  inline void mhattn_forward_check(const functional::Tensor &X,
                                   const functional::Tensor &Y,
                                   const functional::Tensor *context);

}; // class MultiheadAttention

} // namespace model
#endif // !MODEL_MODEL_H
