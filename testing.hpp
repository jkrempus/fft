#include "option_parser.hpp"

struct SizeRange
{
  Int begin;
  Int end;
};

struct SimdImpl { Int val; };

struct Options
{
  bool is_double;
  bool is_bench;
  bool is_real;
  bool is_inverse;
  std::optional<double> precision;
  SimdImpl simd_impl;
  std::string implementation;
  std::vector<SizeRange> size;
};

OptionParser::Result parse_options(int argc, char** argv, Options* dst);

bool run_test(const Options& opt, std::ostream& out);

std::istream& operator>>(std::istream& stream, SimdImpl& simd_impl);
