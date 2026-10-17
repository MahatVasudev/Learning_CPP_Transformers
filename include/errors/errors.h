
#ifndef ERROR_TENSOR_H

#define ERROR_TENSOR_H

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace error {

class ShapeError : public std::invalid_argument {

public:
  explicit ShapeError(std::string operation, std::string expectation,
                      const std::vector<std::size_t> found_shape,
                      std::string additional_message = "");

private:
  std::string build_message_(const std::string &operation,
                             const std::string &expectation,
                             const std::vector<std::size_t> &found_shape,
                             const std::string &addtional_message);
};

class IndexError : public std::out_of_range {
public:
  explicit IndexError(std::string message, const std::size_t found_index,
                      std::string additional_message = "");

private:
  std::string build_message_(const std::string &message,
                             const std::size_t &found_index,
                             const std::string &additional_message);
};

class StateError : public std::logic_error {
public:
  explicit StateError(const std::string &message);
};

template <typename... Args> std::string make_message(const Args &...args) {
  std::ostringstream oss;
  (oss << ... << args);

  return oss.str();
}
std::string format_shape(const std::vector<std::size_t> &shape);
} // namespace error

#endif // !ERROR_TENSOR_H
