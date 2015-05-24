#include "fft_core.h"
#include "fftw3.h"

#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <fstream>
#include <unistd.h>
#include <random>
#include <cstdint>
#include <unordered_set>
#include <sstream>

extern "C" void* valloc(size_t);

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

template<typename T_>
void dump(T_* ptr, Int n, const char* name, ...)
{
  char buf[1 << 10];

  va_list args;
  va_start(args, name);
  vsprintf(buf, name, args);
  va_end(args); 

  std::ofstream(buf, std::ios_base::binary).write((char*) ptr, sizeof(T_) * n);
}

template<typename T>
T* alloc_complex_array(Int n)
{
  return (T*) valloc(2 * n * sizeof(T));
}

template<typename V, ComplexFormat cf, bool is_real> struct TestWrapper { };

template<typename V, ComplexFormat cf>
struct TestWrapper<V, cf, false>
{
  enum { is_real = false };
  VEC_TYPEDEFS(V);
  typedef T value_type;
  State<T>* state;
  T* src;
  T* dst;
  TestWrapper(Int n) :
    state(fft_state<V, cf, cf>(n, valloc(fft_state_memory_size<V>(n)))),
    src((T*) valloc(2 * sizeof(T) * n)),
    dst((T*) valloc(2 * sizeof(T) * n)) { }

  ~TestWrapper()
  {
    free(fft_state_memory_ptr(state));
    free(src);
    free(dst);
  }

  template<typename U>
  void set_input(const U* re, const U* im)
  {
    auto vsrc = (Vec*) src;
    Int vn = state->n / V::vec_size;
    for(Int i = 0; i < vn; i++)
    {
      Complex<Vec> a;
      T* pre = (T*) &a.re;
      T* pim = (T*) &a.im;
      for(Int j = 0; j < V::vec_size; j++)
      {
        pre[j] = T(re[V::vec_size * i + j]);
        pim[j] = T(im[V::vec_size * i + j]);
      }

      store_complex<cf, V>(a, vsrc + i * complex_element_size<cf>(), vn);
    }
  }
 
  void transform() { fft(state, src, dst); }

  template<typename U>
  void get_output(U* re, U* im)
  {
    auto vdst = (Vec*) dst;
    Int vn = state->n / V::vec_size;
    for(Int i = 0; i < vn; i++)
    {
      auto c = load_complex<cf, V>(vdst + i * complex_element_size<cf>(), vn);
      T* pre = (T*) &c.re;
      T* pim = (T*) &c.im;
      for(Int j = 0; j < V::vec_size; j++)
      {
        re[V::vec_size * i + j] = U(pre[j]);
        im[V::vec_size * i + j] = U(pim[j]);
      }
    }
  }
};

template<typename V, ComplexFormat cf>
struct TestWrapper<V, cf, true>
{
  enum { is_real = true };
  VEC_TYPEDEFS(V);
  typedef T value_type;
  RealState<T>* state;
  Int n;
  Int im_offset;
  T* src;
  T* dst;
  TestWrapper(Int n) :
    state(rfft_state<V, cf>(n, valloc(rfft_state_memory_size<V>(n)))),
    n(n),
    im_offset(align_size<T>(n / 2 + 1))
  {
    src = (T*) valloc(n * sizeof(T));
    dst = (T*) valloc(2 * im_offset * sizeof(T));
  }

  ~TestWrapper()
  {
    free(rfft_state_memory_ptr(state));
    free(src);
    free(dst);
  }

  template<typename U>
  void set_input(const U* re, const U* im) { std::copy_n(re, n, src); }
 
  void transform() { rfft(state, src, dst); }

  template<typename U>
  void get_output(U* re, U* im)
  {
    auto vdst = (Vec*) dst;
    for(Int i = 0; i < n / V::vec_size / 2 + 1; i++)
    {
      auto c = load_complex<cf, V>(
        vdst + i * complex_element_size<cf>(), im_offset / V::vec_size);

      T* pre = (T*) &c.re;
      T* pim = (T*) &c.im;
      for(Int j = 0; j < V::vec_size; j++)
      {
        Int k = V::vec_size * i + j;
        if(k <= n / 2)
        {
          re[k] = U(pre[j]);
          im[k] = U(pim[j]);
          if(k > 0)
          {
            re[n - k] = U(pre[j]);
            im[n - k] = -U(pim[j]);
          }
        }
      }
    }
  }
};

template<typename T>
struct InterleavedWrapperBase
{
  Int n;
  T* src;
  T* dst;
  
  InterleavedWrapperBase(Int n) :
    n(n),
    src((T*) valloc(2 * sizeof(T) * n)),
    dst((T*) valloc(2 * sizeof(T) * n)) { }

  template<typename U>
  void set_input(const U* re, const U* im)
  {
    for(Int i = 0; i < n; i++)
    {
      src[2 * i] = T(re[i]);
      src[2 * i + 1] = T(im[i]);
    }
  }

  template<typename U>
  void get_output(U* re, U* im)
  {
    for(Int i = 0; i < n; i++)
    {
      re[i] = U(dst[2 * i]);
      im[i] = U(dst[2 * i + 1]);
    }
  }
};

template<typename T> struct FftwTestWrapper {};

template<> struct FftwTestWrapper<float> : public InterleavedWrapperBase<float>
{
  enum { is_real = false };
  typedef float value_type;
  fftwf_plan plan;

  FftwTestWrapper(Int n) : InterleavedWrapperBase(n)
  {
    plan = fftwf_plan_dft_1d(
      n, (fftwf_complex*) src, (fftwf_complex*) dst,
      FFTW_FORWARD, FFTW_PATIENT);
  }

  ~FftwTestWrapper() { fftwf_destroy_plan(plan); }

  void transform() { fftwf_execute(plan); }
};

template<typename T>
struct ReferenceFft : public InterleavedWrapperBase<T>
{
  enum { is_real = false };
  typedef T value_type;
  fftwf_plan plan;
  Complex<T>* twiddle;
  T* working;

  ReferenceFft(Int n) : InterleavedWrapperBase<T>(n)
  {
    working = new T[2 * n];
    twiddle = new Complex<T>[n / 2];
    auto pi = std::acos(T(-1));
    for(Int i = 0; i < n / 2; i++)
    {
      auto phi = -i * pi / (n / 2);
      twiddle[i] = {std::cos(phi), std::sin(phi)};
    }
  }

  ~ReferenceFft()
  {
    delete twiddle;
    delete working;
  }

  void transform()
  {
    copy(this->src, 2 * this->n, this->dst);

    for(Int dft_size = 1; dft_size < this->n; dft_size *= 2)
    {
      swap(this->dst, working);

      Int twiddle_stride = this->n / 2 / dft_size;
      auto cdst = (Complex<T>*) this->dst;
      auto cworking = (Complex<T>*) working;
      auto ctw = (Complex<T>*) twiddle;
      for(Int i = 0; i < this->n / 2; i += dft_size)
      {
        Int src_i = i;
        Int dst_i = 2 * i;
        Int twiddle_i = 0;
        for(; src_i < i + dft_size;)
        {
          auto a = cworking[src_i];
          auto mul = ctw[twiddle_i] * cworking[src_i + this->n / 2];
          cdst[dst_i] = a + mul;
          cdst[dst_i + dft_size] = a - mul;
          src_i++;
          dst_i++;
          twiddle_i+= twiddle_stride;
        }  
      }
    }
  }
};

template<typename Fft>
void bench(Int n, double requested_operations)
{
  typedef typename Fft::value_type T;
  Fft fft(n);
  T* src = alloc_complex_array<T>(n);
  T* dst = alloc_complex_array<T>(n);
  
  for(Int i = 0; i < n * 2; i++) src[i] = 0.0f;

  double const_part = Fft::is_real ? 2.5 : 5.0;
  auto iter = max<uint64_t>(requested_operations / (const_part * n * log2(n)), 1);
  auto operations = iter * (const_part * n * log2(n));

  double t0 = get_time();
  for(int i = 0; i < iter; i++) fft.transform();
  double t1 = get_time(); 

  printf("%f gflops\n", operations * 1e-9 / (t1 - t0));
}

template<typename Fft0, typename Fft1>
typename Fft0::value_type compare(Int n)
{
  typedef typename Fft0::value_type T;
  Fft0 fft0(n);
  Fft1 fft1(n);
  T* src = alloc_complex_array<T>(n);
  T* dst0 = alloc_complex_array<T>(n);
  T* dst1 = alloc_complex_array<T>(n);
  
  std::mt19937 mt;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for(Int i = 0; i < n * 2; i++) src[i] = dist(mt);
  if(Fft0::is_real || Fft1::is_real)
    for(Int i = n; i < n * 2; i++) src[i] = T(0);

  fft0.set_input(src, src + n);
  fft1.set_input(src, src + n);

  fft0.transform();
  fft1.transform();

  fft0.get_output(dst0, dst0 + n);
  fft1.get_output(dst1, dst1 + n);

#if 0
  dump(fft1.state->twiddle, n, "t.float32");
  dump(fft1.state->state->working, n, "i.float32");
  dump(dst0, 2 * n, "a.float64");
  dump(dst1, 2 * n, "b.float64");
#endif

  auto sum_sumsq = T(0);
  auto diff_sumsq = T(0);
  for(Int i = 0; i < 2 * n; i++)
  {
    sum_sumsq += sq(dst0[i]);
    sum_sumsq += sq(dst1[i]);
    diff_sumsq += sq(dst1[i] - dst0[i]);
  }

  return std::sqrt(diff_sumsq / sum_sumsq);
}

extern "C" void* aligned_alloc(size_t, size_t);

const auto cf = ComplexFormat::split;
#if 0
typedef Scalar<float> V;
#elif defined __ARM_NEON__
typedef Neon V;
#elif defined __AVX__
typedef AvxFloat V;
#elif defined __SSE2__
typedef SseFloat V;
#else
typedef Scalar<float> V;
#endif

template<typename Fft>
void test(Int n)
{
  //TODO: Use long double for ReferenceFft
  printf("difference %e\n", compare<ReferenceFft<double>, Fft>(n));
}

struct Options
{
  std::unordered_set<std::string> flags;
  std::vector<std::string> positional;
};

Options parse_options(int argc, char** argv)
{
  Options r;
  for(Int i = 1; i < argc; i++)
    if(argv[i][0] == '-')
      r.flags.emplace(argv[i]);
    else
      r.positional.emplace_back(argv[i]);

  return r;
}

template<typename Fft>
void test_or_bench3(const Options& opt)
{
  Int log2n;
  std::stringstream(opt.positional[1]) >> log2n;
  Int n = 1 << log2n;

  if(opt.flags.count("-b"))
    bench<Fft>(n, 1e11);
  else
    test<Fft>(n);
}

template<bool is_real, bool is_inverse>
void test_or_bench2(const Options& opt)
{
  if(opt.positional[0] == "fft")
    test_or_bench3<TestWrapper<V, cf, is_real>>(opt);
  else if(opt.positional[0] == "fftw")
    test_or_bench3<FftwTestWrapper<float>>(opt);
  else
    abort();
}

template<bool is_real>
void test_or_bench1(const Options& opt)
{
  if(opt.flags.count("-i"))
    test_or_bench2<is_real, true>(opt);
  else
    test_or_bench2<is_real, false>(opt);
}

void test_or_bench0(const Options& opt)
{
  if(opt.flags.count("-r"))
    test_or_bench1<true>(opt);
  else
    test_or_bench1<false>(opt);
}

int main(int argc, char** argv)
{
  Options opt = parse_options(argc, argv);
  if(opt.positional.size() != 2) abort();
  test_or_bench0(opt);
  return 0;
}
