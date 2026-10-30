#ifndef TENSOR_FUNCTIONAL_H

#define TENSOR_FUNCTIONAL_H
#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
namespace functional {

// TODO: Need to add support for multiple data types (only numeric)
// like float, int, double std::size_t... do not support string, char etc

// Tensor Type
class Tensor {
public:
  explicit Tensor(std::vector<std::size_t> shape);

  // NOTE: rule of 5
  ~Tensor();                                  // Destructor
  Tensor(const Tensor &other);                // Copy
  Tensor &operator=(const Tensor &other);     // Copy assignment
  Tensor(Tensor &&other) noexcept;            // Move
  Tensor &operator=(Tensor &&other) noexcept; // Move assignment

  // GET Methods
  const std::vector<std::size_t> &shape() const { return shape_; }
  const std::vector<std::size_t> &strides() const { return strides_; }
  std::size_t rank() const { return shape_.size(); }
  std::size_t numel() const { return numel_; }
  float *data() { return data_.get(); }
  const float *data() const { return data_.get(); }

  // Shaping Methods
  void reshape(std::vector<std::size_t> shape);
  void squeeze(int &dim);
  void unsqueeze(int &dim);
  Tensor squeeze_copy(int &dim) const;
  Tensor unsqueeze_copy(int &dim) const;
  Tensor view(std::vector<std::size_t> shape) const;
  Tensor permute(std::vector<std::size_t> perm) const;
  Tensor transpose(int dim0, int dim1) const;
  Tensor broadcast_to(std::vector<std::size_t> shape) const;
  Tensor broadcast_like(const Tensor &t) const;

  // Indexing
  template <typename... Idx> float &operator()(Idx... idx) {
    return data_[offset(idx...)];
  }
  template <typename... Idx> const float &operator()(Idx... idx) const {
    return data_[offset(idx...)];
  }

  // Operator

  // TODO: need to make the alternative of these operations
  // currently they only take the order TENSOR + FLOAT
  // not FLOAT + TENSOR, even though they are the same thing.

  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator+(const float &other) const;
  Tensor operator*(const float &other) const;
  Tensor operator/(const float &other) const;

  // Operation methods

  void add(const Tensor &other);
  void add(const float &other);
  void multiply(const float &other);
  void divide(const float &other);

  // Reducer Operations

  Tensor sum(int axis, bool keepdims = false) const;
  Tensor sum(std::vector<int> &axes, bool keepdims = false) const;
  Tensor norm(std::size_t norm, std::vector<int> &axes,
              bool keepdims = false) const;

  // For each method for reduction methods
  //
  // Callback signature:
  // void (std::size_t input_offset,
  // std::size_t output_offset,
  // std::size_t reduced_coord);
  template <typename Callback>
  static void for_each(const std::vector<std::size_t> &input_shape,
                       const std::vector<std::size_t> &input_strides,
                       const std::vector<std::size_t> &out_strides,
                       std::size_t axis, bool keepdims, Callback &&callback) {
    const std::size_t rank_ = input_shape.size();
    std::size_t numel = 1;
    for (std::size_t dim : input_shape) {
      numel *= dim;
    }

    for (std::size_t linear = 0; linear < numel; linear++) {
      std::size_t remainder = linear;
      std::size_t input_offset = 0;
      std::size_t output_offset = 0;
      std::size_t reduced_coord = 0;
      for (std::size_t d = rank_; d-- > 0;) {
        const std::size_t coords = remainder % input_shape[d];
        remainder /= input_shape[d];

        input_offset += coords * input_strides[d];

        if (d == axis) {
          reduced_coord = coords;
          continue;
        }

        const std::size_t out_d = keepdims || d < axis ? d : d - 1;

        output_offset += coords * out_strides[out_d];
      }

      callback(input_offset, output_offset, reduced_coord);
    }
  }

  // From Methods
  static Tensor from_values(std::vector<std::size_t> shape,
                            const std::vector<float> &values);
  static Tensor from_values(std::vector<std::size_t> shape,
                            std::initializer_list<float> values);
  static Tensor from_pointer(std::vector<std::size_t> shape, const float *src,
                             const std::size_t count);

  // Misc.
  void fill(float value);

  Tensor clone() const;
  bool is_contiguous() const;
  static bool
  is_broadcast_possible(const std::vector<std::size_t> &got_shape,
                        const std::vector<std::size_t> &target_shape) noexcept;

  static std::vector<std::size_t>
  broadcast_shape(const std::vector<std::size_t> &shape_a,
                  const std::vector<std::size_t> &shape_b);

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

  // Private constructor, can break if added without check
  Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape,
         std::vector<std::size_t> strides, std::size_t numel) noexcept;

  static std::vector<std::size_t>
  compute_strides_(std::vector<std::size_t> &shape, std::size_t &numel);
  static std::vector<std::size_t>
  broadcast_strides(const std::vector<std::size_t> &shape,
                    const std::vector<std::size_t> &input_strides,
                    const std::vector<std::size_t> &outpust_shape);

  static std::vector<std::size_t>
  reduction_size(const std::vector<std::size_t> &shape, int axis,
                 bool keepdims);
  static std::vector<std::size_t>
  reduction_size(const std::vector<std::size_t> &shape, std::vector<int> &axis,
                 bool keepdims);

  // static std::size_t unravel_index(std::size_t linear,
  //                                  const std::vector<std::size_t> &shape);

  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
  std::size_t numel_;
  std::shared_ptr<float[]> data_;
}; // class Tensor

void require_contiguous(const Tensor &t, const std::string &operation = "",
                        const std::string &variable_name = "");

Tensor reshape(Tensor &t, std::vector<std::size_t> &shape);
} // namespace functional
#endif // !TENSOR_FUNCTIONAL_H
