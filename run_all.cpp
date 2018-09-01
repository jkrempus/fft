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
  if(opt.is_inverse) s << " -i";
  if(opt.is_bench) s << " -b";
 
  return s.str();
}

int run(
  const std::vector<Int>& size, std::mutex& output_mutex, Int verbosity,
  Int simd_impl)
{
  for(auto [is_real, is_inverse] : {
    std::make_pair(true, true),
    std::make_pair(true, false),
    std::make_pair(false, true),
    std::make_pair(false, false)})
  {
    Options opt;
    opt.implementation = "fft";
    for(auto e : size) opt.size.push_back({e, e + 1});

    opt.precision = 1.2e-7;
    opt.is_bench =  false;
    opt.is_real = is_real;
    opt.is_inverse = is_inverse;
    opt.simd_impl = {simd_impl};

    if(verbosity >= 2)
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

    if(verbosity >= 3)
    {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cerr << s.str() << std::endl;
    }
  }
}

void run(
  const std::vector<std::vector<Int>>& sizes, Int num_threads, Int verbosity,
  Int simd_impl)
{
  int64_t total_size = 0;
  for(auto& s : sizes) total_size += num_processed_elements(s);

  std::mutex mutex;
  std::vector<std::future<void>> futures;
 
  int64_t done_size = 0;
  std::atomic<Int> size_idx = 0;

  for(Int i = 0; i < num_threads; i++)
    futures.push_back(std::async(
      std::launch::async,
      [&]()
    {
      while(true)
      {
        int idx = size_idx++;
        if(idx >= sizes.size()) return;
        auto& s = sizes[idx];

        run(s, mutex, verbosity, simd_impl);

        std::lock_guard<std::mutex> lock(mutex);
        done_size += num_processed_elements(s);

        if(verbosity >= 1)
          std::cerr
            << "Done " << (100.0 * done_size / total_size) << "%" << std::endl;
      }
    }));

  for(auto& f : futures) f.get();
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
  Int max_size, Int max_dim, int64_t total_size,
  std::vector<std::vector<Int>>& sizes)
{
  static std::set<std::vector<Int>> present_sizes;

  int64_t done_size = 0;

  for(auto& s : sizes)
  {
    present_sizes.insert(s);
    done_size += num_processed_elements(s);
  }

  std::vector<std::vector<Int>> all_sizes;
  get_all_sizes(std::vector<Int>{}, max_size, max_dim, all_sizes);
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
      done_size += num_processed_elements(s);
    }
  }

  return double(num_tested) / double(all_sizes.size());
}

int main(int argc, char** argv)
{
  Int max_size = 22;
  Int max_dim = 5;
  Int verbosity = 1;
  Int threads = 1;
  Int total = 1000000000;
  SimdImpl simd_impl = {0};

  OptionParser op;
  op.add_optional_flag("-v", "Verbosity level.", &verbosity);
  op.add_optional_flag("--threads", "Number of threads.", &threads);
  op.add_optional_flag("-d", "Maximal number of dimensions.", &max_dim);
  op.add_optional_flag(
    "--simd", "which SIMD implementation to use.", &simd_impl);
  op.add_optional_flag(
    "-s", "Maximal value for the binary logarithm of transform size.",
    &max_size);
  op.add_optional_flag(
    "-t", "The total number of elements that will be processed in all of the tests.",
    &total);

  op.parse(argc, argv);

  std::vector<std::vector<Int>> sizes;
  for(Int i = 1; i < max_size + 1; i++) sizes.push_back({i});
  for(Int i = 1; i < max_size / 2 + 1; i++) sizes.push_back({i, i});
  for(Int i = 1; i < max_size / 3 + 1; i++) sizes.push_back({i, i, i});
  double tested_percentage = add_random_sizes(
    max_size, max_dim, total, sizes);

  if(verbosity >= 1)
    std::cerr
      << (100.0 * tested_percentage) << "% of all possible sizes with up to "
      << max_dim << " dimensions " << std::endl
      << "and number of elements up to 2^"
      << max_size << " will be tested." << std::endl;

  run(sizes, threads, verbosity, simd_impl.val);

  return 0;
}
