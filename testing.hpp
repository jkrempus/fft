#include <string>
#include <optional>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <functional>

using Int = ptrdiff_t;
using Uint = size_t;

struct SizeRange
{
  Int begin;
  Int end;
};

struct Options
{
  bool is_bench;
  bool is_real;
  bool is_inverse;
  std::optional<double> precision;
  std::string implementation;
  std::vector<SizeRange> size;
};

//TODO add --help flag handling
class OptionParser
{
public:
  struct Result
  {
    bool data_valid;
    bool error;
    std::string message;
    operator bool() const { return data_valid; }
  };

  struct Switch
  {
    std::string description;
    bool* dst;
  };

  struct Option
  {
    std::string name;
    std::string description;
    std::function<bool(const char*)> handler;
    Int min_num = 0;
    Int max_num = 1;
    Int num = 0;
  };

  void add_switch(
    const std::string_view& name, const std::string_view& description,
    bool* dst);

  template<typename T>
  void add_optional_flag(
    const std::string_view& name, const std::string_view& description,
    std::optional<T>* dst);

  template<typename T>
  void add_positional(
    const std::string_view& name, const std::string_view& description,
    T* dst);

  template<typename T>
  void add_multi_positional(
    const std::string_view& name, const std::string_view& description,
    Int min_num, Int max_num,
    std::vector<T>* dst);

  Result parse(int argc, char** argv);

private:  
  template<typename... Args>
  static Result fail(const Args&... args);

  std::unordered_map<std::string, Switch> switches;
  std::unordered_map<std::string, Option> flags;
  std::vector<Option> positional;
};

OptionParser::Result parse_options(int argc, char** argv, Options* dst);

bool run_test(const Options& opt, std::ostream& out);
