#include "option_parser.hpp"

struct SizeRange
{
  Int begin;
  Int end;
};

struct SimdImpl { Int val = 0; };

struct Options
{
  bool is_double = false;
  bool is_bench = false;
  bool is_real = false;
  bool is_inverse = false;
  std::optional<double> precision;
  double num_ops = 0.0;
  SimdImpl simd_impl;
  ptrdiff_t misaligned_by = 0;
  std::string implementation;
  std::vector<SizeRange> size;
};

OptionParser::Result parse_options(int argc, char** argv, Options* dst);

bool run_test(const Options& opt, std::ostream& out);

std::istream& operator>>(std::istream& stream, SimdImpl& simd_impl);
