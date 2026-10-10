#ifndef TENSOR_FUNCTIONAL_H

#define TENSOR_FUNCTIONAL_H
#include <cassert>
#include <vector>
namespace functional {

class Tensor {
public:
  explicit Tensor(std::vector<std::size_t> shape);

  // NOTE: rule of 5
  ~Tensor();                                  // Destructor
  Tensor(const Tensor &other);                // Copy
  Tensor &operator=(const Tensor &other);     // Copy assignment
  Tensor(Tensor &&other) noexcept;            // Move
  Tensor &operator=(Tensor &&other) noexcept; // Move assignment

  const std::vector<std::size_t> &shape() const { return shape_; }
  const std::vector<std::size_t> &strides() const { return strides_; }

  std::size_t rank() const { return shape_.size(); }
  std::size_t numel() const { return numel_; }

  float *data() { return data_; }

  const float *data() const { return data_; }

  template <typename... Idx> float &operator()(Idx... idx) {
    return data_[offset(idx...)];
  }

  template <typename... Idx> const float &operator()(Idx... idx) const {
    return data_[offset(idx...)];
  }

  void fill(float value);

  Tensor clone() const;

private:
  template <typename... Idx> std::size_t offset(Idx... idx) const {
    static_assert(sizeof...(Idx) > 0, "need at least one index");
    assert(sizeof...(Idx) == shape_.size() && "index count must equal rank");
    const std::size_t indices[] = {static_cast<std::size_t>(idx)...};
    std::size_t off = 0;
    for (std::size_t d = 0; d < shape_.size(); ++d) {
      assert(indices[d] < shape_[d] && "index out of bounds");
      off += indices[d] * strides_[d];
    }
    return off;
  }

  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
  std::size_t numel_;
  float *data_;
};
} // namespace functional
#endif // !TENSOR_FUNCTIONAL_H
