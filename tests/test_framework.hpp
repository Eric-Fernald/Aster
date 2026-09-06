#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace aster_test {

struct Case {
  const char* name;
  std::function<void()> fn;
};

struct Registry {
  static std::vector<Case>& cases() { static std::vector<Case> c; return c; }
  static int& failures() { static int f = 0; return f; }
};

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { Registry::cases().push_back({name, std::move(fn)}); }
};

struct Failure { std::string msg; };

std::string TempDir(const std::string& tag);

}  // namespace aster_test

#define ASTER_TEST(name)                                   \
  static void aster_test_##name();                         \
  static ::aster_test::Registrar aster_reg_##name(#name, aster_test_##name); \
  static void aster_test_##name()

#define ASTER_FAIL(msg)                                                              \
  do {                                                                               \
    std::ostringstream _os;                                                          \
    _os << __FILE__ << ":" << __LINE__ << ": " << msg;                               \
    throw ::aster_test::Failure{_os.str()};                                          \
  } while (0)

#define ASTER_CHECK(cond) do { if (!(cond)) ASTER_FAIL("check failed: " #cond); } while (0)
#define ASTER_CHECK_EQ(a, b) do { auto _a = (a); auto _b = (b); if (!(_a == _b)) ASTER_FAIL(#a " == " #b " (" << _a << " vs " << _b << ")"); } while (0)
#define ASTER_CHECK_NEAR(a, b, eps) do { double _a = (a), _b = (b); if (std::fabs(_a - _b) > (eps)) ASTER_FAIL(#a " ~= " #b " (" << _a << " vs " << _b << ")"); } while (0)
#define ASTER_CHECK_OK(expr) do { auto _s = (expr); if (!_s.ok()) ASTER_FAIL(#expr " -> " << _s.ToString()); } while (0)
#define ASTER_ASSIGN_OK(lhs, expr) auto ASTER_CONCAT(_t_, __LINE__) = (expr); if (!ASTER_CONCAT(_t_, __LINE__).ok()) ASTER_FAIL(#expr " -> " << ASTER_CONCAT(_t_, __LINE__).status().ToString()); lhs = std::move(ASTER_CONCAT(_t_, __LINE__)).value();
