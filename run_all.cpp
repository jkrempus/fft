#include "testing.hpp"

#include <iostream>
#include <sstream>

std::string format_option(const Options& opt)
{
  std::stringstream s;
  s << "run " << opt.implementation;
  for(auto& e : opt.size)
    if(e.end - e.begin == 1)
      s << " " << e.begin;
    else
      s << " " << e.begin << "-" << e.end;

  if(opt.precision)
    s << " -p" << *opt.precision;

  if(opt.is_real) s << " -r";
  if(opt.is_inverse) s << " -i";
  if(opt.is_bench) s << " -b";
 
  return s.str();
}

int run_single(const std::vector<Int>& size, bool is_real, bool is_inverse)
{
  Options opt;
  opt.implementation = "fft";
  for(auto e : size) opt.size.push_back({e, e + 1});

  opt.precision = 1.2e-7;
  opt.is_bench =  false;
  opt.is_real = is_real;
  opt.is_inverse = is_inverse;

  std::cerr << format_option(opt) << std::endl;

  std::stringstream s;
  if(!run_test(opt, s))
  {
    std::cerr << s.str() << std::endl;
    exit(1);
  }
}

int run_with_size(const std::vector<Int>& size)
{
  run_single(size, false, false);
  run_single(size, false, true);
  run_single(size, true, false);
  run_single(size, true, true);
}

int main(int argc, char** argv)
{
  Int m = 22;

  for(Int i = 1; i < m + 1; i++) run_with_size({i});
  for(Int i = 1; i < m / 2 + 1; i++) run_with_size({i, i});
  for(Int i = 1; i < m / 3 + 1; i++) run_with_size({i, i, i});

  return 0;
}
