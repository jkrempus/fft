#include "testing.hpp"

#include <iostream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <random>

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
    s << " -p " << *opt.precision;

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
  static std::set<std::vector<Int>> already_run;

  if(already_run.count(size) == 1) return 0;
  already_run.insert(size);

  run_single(size, false, false);
  run_single(size, false, true);
  run_single(size, true, false);
  run_single(size, true, true);

  return 1;
}

void get_all_sizes(
  const std::vector<Int>& s, Int sz_left, Int max_len,
  std::vector<std::vector<Int>>& dst)
{
  if(s.size() > 0) dst.push_back(s);
  if(s.size() < max_len)
    for(Int i = 1; i < sz_left + 1; i++)
    {
      std::vector<Int> new_s = s;
      new_s.push_back(i);
      get_all_sizes(new_s, sz_left - i, max_len, dst);
    }
}

std::vector<double> get_weights(
  const std::vector<std::vector<Int>>& all_sizes)
{
  auto key = [=](const std::vector<Int>& a)
  {
    return std::pair<Int, Int>(
      a.size(),
      std::accumulate(a.begin(), a.end(), Int(0)));
  };

  std::map<std::pair<Int, Int>, Int> counts;
  for(auto& e : all_sizes) counts[key(e)]++;

  std::vector<double> r; 
  for(auto& e : all_sizes) r.push_back(1.0 / double(counts[key(e)]));
  return r;
}

void run_random_sizes(Int m, Int max_dim)
{
  std::vector<std::vector<Int>> all_sizes;
  get_all_sizes(std::vector<Int>{}, m, max_dim, all_sizes);
  auto w = get_weights(all_sizes);
  std::mt19937 rng;
  std::discrete_distribution<Int> dist{w.begin(), w.end()};

  Int num_tested = 0;
  int64_t total_size = 1000000000;
  int64_t done_size = 0;

  while(done_size < total_size && num_tested != all_sizes.size())
  {
    auto& s = all_sizes[dist(rng)];
    if(run_with_size(s) > 0)
    {
      num_tested++;
      done_size += int64_t(1) << std::accumulate(s.begin(), s.end(), Int(0));
      std::cerr
        << "Done " << (100.0 * done_size / total_size)
        << "%. Tested " << (100.0 * num_tested / all_sizes.size())
        << "% of all possible sizes." << std::endl;
    }
  }

  std::cerr << all_sizes.size() << std::endl;
}

int main(int argc, char** argv)
{
  Int m = 22;
  for(Int i = 1; i < m + 1; i++) run_with_size({i});
  for(Int i = 1; i < m / 2 + 1; i++) run_with_size({i, i});
  for(Int i = 1; i < m / 3 + 1; i++) run_with_size({i, i, i});
  run_random_sizes(m, 5);
  return 0;
}
