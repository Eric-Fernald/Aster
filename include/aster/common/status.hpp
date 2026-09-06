#pragma once
#include <string>
#include <utility>
#include <variant>

namespace aster {

enum class ErrorCode { Ok, Invalid, NotSupported, OutOfMemory, IoError, NotFound, Internal, Corrupt };

class Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string msg) : code_(code), msg_(std::move(msg)) {}
  static Status OK() { return {}; }
  static Status Invalid(std::string m) { return {ErrorCode::Invalid, std::move(m)}; }
  static Status NotSupported(std::string m) { return {ErrorCode::NotSupported, std::move(m)}; }
  static Status OutOfMemory(std::string m) { return {ErrorCode::OutOfMemory, std::move(m)}; }
  static Status IoError(std::string m) { return {ErrorCode::IoError, std::move(m)}; }
  static Status NotFound(std::string m) { return {ErrorCode::NotFound, std::move(m)}; }
  static Status Internal(std::string m) { return {ErrorCode::Internal, std::move(m)}; }
  static Status Corrupt(std::string m) { return {ErrorCode::Corrupt, std::move(m)}; }

  bool ok() const { return code_ == ErrorCode::Ok; }
  ErrorCode code() const { return code_; }
  const std::string& message() const { return msg_; }
  std::string ToString() const;

 private:
  ErrorCode code_ = ErrorCode::Ok;
  std::string msg_;
};

template <typename T>
class Result {
 public:
  Result(T v) : v_(std::move(v)) {}
  Result(Status s) : v_(std::move(s)) {}
  bool ok() const { return std::holds_alternative<T>(v_); }
  Status status() const { return ok() ? Status::OK() : std::get<Status>(v_); }
  T& value() & { return std::get<T>(v_); }
  T&& value() && { return std::get<T>(std::move(v_)); }
  const T& value() const& { return std::get<T>(v_); }
  T& operator*() { return value(); }
  T* operator->() { return &value(); }

 private:
  std::variant<T, Status> v_;
};

}  // namespace aster

#define ASTER_RETURN_NOT_OK(expr)  \
  do {                             \
    ::aster::Status _s = (expr);   \
    if (!_s.ok()) return _s;       \
  } while (0)

#define ASTER_CONCAT_(a, b) a##b
#define ASTER_CONCAT(a, b) ASTER_CONCAT_(a, b)
#define ASTER_ASSIGN_OR_RETURN(lhs, expr)                                                       \
  auto ASTER_CONCAT(_aster_r_, __LINE__) = (expr);                                              \
  if (!ASTER_CONCAT(_aster_r_, __LINE__).ok()) return ASTER_CONCAT(_aster_r_, __LINE__).status(); \
  lhs = std::move(ASTER_CONCAT(_aster_r_, __LINE__)).value();
