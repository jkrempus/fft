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

void run(const std::vector<Int>& size)
{
  run_single(size, false, false);
  run_single(size, false, true);
  run_single(size, true, false);
  run_single(size, true, true);
}

void run(const std::vector<std::vector<Int>>& sizes)
{
  int64_t total_size = 0;
  for(auto& s : sizes)
    total_size += int64_t(1) << std::accumulate(s.begin(), s.end(), Int(0));

  int64_t done_size = 0;
  for(auto& s : sizes)
  {
    run(s);
    done_size += int64_t(1) << std::accumulate(s.begin(), s.end(), Int(0));
    std::cerr << "Done " << (100.0 * done_size / total_size) << "%" << std::endl;
  }
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

double add_random_sizes(
  Int max_total, Int max_dim, int64_t total_size,
  std::vector<std::vector<Int>>& sizes)
{
  static std::set<std::vector<Int>> present_sizes;

  int64_t done_size = 0;

  for(auto& s : sizes)
  {
    present_sizes.insert(s);
    done_size += int64_t(1) << std::accumulate(s.begin(), s.end(), Int(0));
  }

  std::vector<std::vector<Int>> all_sizes;
  get_all_sizes(std::vector<Int>{}, max_total, max_dim, all_sizes);
  auto w = get_weights(all_sizes);

  Int num_tested = 0;
  for(auto& s : all_sizes)
    if(present_sizes.count(s) == 1)
      num_tested++;

  std::mt19937 rng;
  std::discrete_distribution<Int> dist{w.begin(), w.end()};

  while(done_size < total_size && num_tested != all_sizes.size())
  {
    auto& s = all_sizes[dist(rng)];
    if(present_sizes.count(s) == 0)
    {
      present_sizes.insert(s);
      sizes.push_back(s);
      num_tested++;
      done_size += int64_t(1) << std::accumulate(s.begin(), s.end(), Int(0));
    }
  }

  return double(num_tested) / double(all_sizes.size());
}

int main(int argc, char** argv)
{
  Int max_total = 22;
  Int max_dim = 5;

  std::vector<std::vector<Int>> sizes;
  for(Int i = 1; i < max_total + 1; i++) sizes.push_back({i});
  for(Int i = 1; i < max_total / 2 + 1; i++) sizes.push_back({i, i});
  for(Int i = 1; i < max_total / 3 + 1; i++) sizes.push_back({i, i, i});
  double tested_percentage =
    add_random_sizes(max_total, max_dim, 1000000000, sizes);

  std::cerr
    << (100.0 * tested_percentage) << "% of all possible sizes with up to "
    << max_dim << " dimensions \nand total number of elements up to 2^"
    << max_total << " will be tested." << std::endl;

  run(sizes);

  return 0;
}
