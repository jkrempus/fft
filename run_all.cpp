#include "testing.hpp"

#include <iostream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <random>
#include <mutex>
#include <future>

struct RunAllOptions
{
  Int max_size = 22;
  Int max_dim = 5;
  Int verbosity = 1;
  Int threads = 1;
  Int total = 1000000000;
  std::string implementation;
  SimdImpl simd_impl;
  bool random_alignment = false;
  bool is_double = false;
};

int64_t num_processed_elements(const std::vector<Int>& a)
{
  return 4 * (int64_t(1) << std::accumulate(a.begin(), a.end(), Int(0)));
}

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
  if(opt.is_double) s << " -d";
  if(opt.is_inverse) s << " -i";
  if(opt.is_bench) s << " -b";
  if(opt.misaligned_by != 0) s << " --misaligned-by " << opt.misaligned_by;
 
  return s.str();
}

void run(
  const std::vector<Int>& size, Int misaligned_by,
  std::mutex& output_mutex, const RunAllOptions& ra_opt)
{
  for(auto [is_real, is_inverse] : {
    std::make_pair(true, true),
    std::make_pair(true, false),
    std::make_pair(false, true),
    std::make_pair(false, false)})
  {
    Options opt;
    opt.implementation = std::string(ra_opt.implementation);
    for(auto e : size) opt.size.push_back({e, e + 1});

    opt.precision = 1.2e-7;
    opt.is_double = ra_opt.is_double;
    opt.is_bench =  false;
    opt.is_real = is_real;
    opt.is_inverse = is_inverse;
    opt.simd_impl = {ra_opt.simd_impl};
    opt.misaligned_by = misaligned_by;

    if(ra_opt.verbosity >= 2)
    {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cerr << format_option(opt) << std::endl;
    }

    std::stringstream s;
    if(!run_test(opt, s))
    {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cerr
        << "Error while running test \""
        << format_option(opt) << "\":" << std::endl
        << s.str() << std::endl;

      exit(1);
    }

    if(ra_opt.verbosity >= 3)
    {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cerr << s.str() << std::endl;
    }
  }
}

void run(const std::vector<std::vector<Int>>& sizes, const RunAllOptions& opt)
{
  int64_t total_size = 0;
  for(auto& s : sizes) total_size += num_processed_elements(s);

  std::mt19937 rng;
  std::uniform_int_distribution<Int> dist(0, 4095);

  std::vector<Int> misaligned_by(sizes.size(), 0);
  if(opt.random_alignment) for(auto& e: misaligned_by) e = dist(rng);

  std::mutex mutex;
  std::vector<std::future<void>> futures;
 
  int64_t done_size = 0;
  std::atomic<Int> size_idx = 0;

  auto fn = [&]()
  {
    while(true)
    {
      int idx = size_idx++;
      if(idx >= sizes.size()) return;
      auto& s = sizes[idx];

      run(s, misaligned_by[idx], mutex, opt);

      std::lock_guard<std::mutex> lock(mutex);
      done_size += num_processed_elements(s);

      if(opt.verbosity >= 1)
        std::cerr
          << "Done " << (100.0 * done_size / total_size) << "%" << std::endl;
    }
  };

  if(opt.threads == 1)
    fn();
  else
  {
    for(Int i = 0; i < opt.threads; i++)
      futures.push_back(std::async(std::launch::async, fn));

    for(auto& f : futures) f.get();
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
  const RunAllOptions& opt, std::vector<std::vector<Int>>& sizes)
{
  static std::set<std::vector<Int>> present_sizes;

  int64_t done_size = 0;

  for(auto& s : sizes)
  {
    present_sizes.insert(s);
    done_size += num_processed_elements(s);
  }

  std::vector<std::vector<Int>> all_sizes;
  get_all_sizes(std::vector<Int>{}, opt.max_size, opt.max_dim, all_sizes);
  auto w = get_weights(all_sizes);

  Int num_tested = 0;
  for(auto& s : all_sizes)
    if(present_sizes.count(s) == 1)
      num_tested++;

  std::mt19937 rng;
  std::discrete_distribution<Int> dist{w.begin(), w.end()};

  while(done_size < opt.total && num_tested != all_sizes.size())
  {
    auto& s = all_sizes[dist(rng)];
    if(present_sizes.count(s) == 0)
    {
      present_sizes.insert(s);
      sizes.push_back(s);
      num_tested++;
      done_size += num_processed_elements(s);
    }
  }

  return double(num_tested) / double(all_sizes.size());
}

int main(int argc, char** argv)
{
  RunAllOptions opt;
  OptionParser parser;
  parser.add_optional_flag("-v", "Verbosity level.", &opt.verbosity);
  parser.add_optional_flag("--threads", "Number of threads.", &opt.threads);
  parser.add_optional_flag("-d", "Maximal number of dimensions.", &opt.max_dim);
  parser.add_optional_flag(
    "--simd", "which SIMD implementation to use.", &opt.simd_impl);
  parser.add_optional_flag(
    "-s", "Maximal value for the binary logarithm of transform size.",
    &opt.max_size);
  parser.add_optional_flag(
    "-t", "The total number of elements that will be processed in all of the tests.",
    &opt.total);
  parser.add_optional_flag(
    "--misaligned_by", "which SIMD implementation to use.", &opt.simd_impl);
  parser.add_switch(
    "--random-alignment", "Use randomly aligned buffers.", &opt.random_alignment);
  parser.add_switch("--is-double", "Use double precision.", &opt.is_double);

  parser.add_positional(
    "implementation", "Fft implementation to test.", &opt.implementation);

  auto parsing_result = parser.parse(argc, argv);
  if(!parsing_result)
  {
    std::cerr << parsing_result.message << std::endl;
    return parsing_result.error ? 1 : 0;
  }

  std::vector<std::vector<Int>> sizes;
  for(Int i = 1; i < opt.max_size + 1; i++) sizes.push_back({i});
  for(Int i = 1; i < opt.max_size / 2 + 1; i++) sizes.push_back({i, i});
  for(Int i = 1; i < opt.max_size / 3 + 1; i++) sizes.push_back({i, i, i});
  double tested_percentage = add_random_sizes(opt, sizes);

  if(opt.verbosity >= 1)
    std::cerr
      << (100.0 * tested_percentage) << "% of all possible sizes with up to "
      << opt.max_dim << " dimensions " << std::endl
      << "and number of elements up to 2^"
      << opt.max_size << " will be tested." << std::endl;

  run(sizes, opt);

  return 0;
}
