#include <immintrin.h>

typedef long Int;

float fsin[64] = {
  0,
  0x1p+0,
  0x1.6a09e6p-1,
  0x1.87de2cp-2,
  0x1.8f8b84p-3,
  0x1.917a6cp-4,
  0x1.91f66p-5,
  0x1.92156p-6,
  0x1.921d2p-7,
  0x1.921f1p-8,
  0x1.921f8cp-9,
  0x1.921facp-10,
  0x1.921fb4p-11,
  0x1.921fb6p-12,
  0x1.921fb6p-13,
  0x1.921fb6p-14,
  0x1.921fb6p-15,
  0x1.921fb6p-16,
  0x1.921fb6p-17,
  0x1.921fb6p-18,
  0x1.921fb6p-19,
  0x1.921fb6p-20,
  0x1.921fb6p-21,
  0x1.921fb6p-22,
  0x1.921fb6p-23,
  0x1.921fb6p-24,
  0x1.921fb6p-25,
  0x1.921fb6p-26,
  0x1.921fb6p-27,
  0x1.921fb6p-28,
  0x1.921fb6p-29,
  0x1.921fb6p-30,
  0x1.921fb6p-31,
  0x1.921fb6p-32,
  0x1.921fb6p-33,
  0x1.921fb6p-34,
  0x1.921fb6p-35,
  0x1.921fb6p-36,
  0x1.921fb6p-37,
  0x1.921fb6p-38,
  0x1.921fb6p-39,
  0x1.921fb6p-40,
  0x1.921fb6p-41,
  0x1.921fb6p-42,
  0x1.921fb6p-43,
  0x1.921fb6p-44,
  0x1.921fb6p-45,
  0x1.921fb6p-46,
  0x1.921fb6p-47,
  0x1.921fb6p-48,
  0x1.921fb6p-49,
  0x1.921fb6p-50,
  0x1.921fb6p-51,
  0x1.921fb6p-52,
  0x1.921fb6p-53,
  0x1.921fb6p-54,
  0x1.921fb6p-55,
  0x1.921fb6p-56,
  0x1.921fb6p-57,
  0x1.921fb6p-58,
  0x1.921fb6p-59,
  0x1.921fb6p-60,
  0x1.921fb6p-61,
  0x1.921fb6p-62};

float fcos[64] = {
  -0x1p+0,
  -0x1.777a5cp-25,
  0x1.6a09e6p-1,
  0x1.d906bcp-1,
  0x1.f6297cp-1,
  0x1.fd88dap-1,
  0x1.ff621ep-1,
  0x1.ffd886p-1,
  0x1.fff622p-1,
  0x1.fffd88p-1,
  0x1.ffff62p-1,
  0x1.ffffd8p-1,
  0x1.fffff6p-1,
  0x1.fffffep-1,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0,
  0x1p+0};

const float pi = 0x1.921fb6p+1f;
typedef float T;
typedef __m256 Vec;

template<typename First, typename Second>
struct pair
{
  First first;
  Second second;
};

template<typename T_>
struct ComplexPtrs
{
  T_* re;
  T_* im;

  ComplexPtrs& operator+=(Int offset)
  {
    re += offset;
    im += offset;
    return *this;
  }
  
  ComplexPtrs& operator-=(Int offset)
  {
    re -= offset;
    im -= offset;
    return *this;
  }

  ComplexPtrs operator+(Int offset) { return {re + offset, im + offset}; }
  ComplexPtrs operator-(Int offset) { return {re - offset, im - offset}; }
};

pair<Vec, Vec> transpose_128(Vec a, Vec b)
{
  return {
    _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 2, 0, 0)),
    _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 3, 0, 1)) };
}

template<Int elements_per_vec>
pair<Vec, Vec> interleave(Vec a, Vec b)
{
  if(elements_per_vec == 8)
    return transpose_128(_mm256_unpacklo_ps(a, b), _mm256_unpackhi_ps(a, b));
  else if(elements_per_vec == 4)
  {
    return transpose_128(
      _mm256_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 1, 0)), 
      _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 3, 2)));
  }
  else if (elements_per_vec == 2)
    return transpose_128(a, b);
}

void init_twiddle(Int len, ComplexPtrs<T> dst)
{
  auto end = dst + len;
  Int table_index = 0;
  end.re[-1] = T(0);
  end.im[-1] = T(0);
  end.re[-2] = T(1);
  end.im[-2] = T(0);

  for(Int size = 2; size < len; size *= 2)
  {
    table_index++;
    auto c = fcos[table_index];
    auto s = fsin[table_index];

    auto prev = end - size;
    auto current = end - 2 * size;
    for(Int j = 0; j < size / 2; j++)
    {
      T prev_re = prev.re[j];
      T prev_im = prev.im[j];
      current.re[2 * j] = prev_re;
      current.im[2 * j] = prev_im;
      current.re[2 * j + 1] = prev_re * c + prev_im * s;
      current.im[2 * j + 1] = prev_im * c - prev_re * s;
    }
  }
}

template<typename T>
void swap(T& a, T& b)
{
  T tmpa = a;
  T tmpb = b;
  a = tmpb;
  b = tmpa;
}

extern "C" int sprintf(char* s, const char* fmt, ...);
template<typename T_> void dump(T_* ptr, Int n, const char* name);

template<Int dft_size>
void ct_dft_size_pass(
  Int n, ComplexPtrs<T> src, ComplexPtrs<T> twiddle, ComplexPtrs<T> dst)
{
  const Int vec_size = sizeof(Vec) / sizeof(T);
  Int vn = n / vec_size;
  auto vsrc0_re = (Vec*) src.re;
  auto vsrc0_im = (Vec*) src.im;
  auto vsrc1_re = (Vec*) src.re + vn / 2;
  auto vsrc1_im = (Vec*) src.im + vn / 2;
  auto vdst_re = (Vec*) dst.re;
  auto vdst_im = (Vec*) dst.im;
  if(dft_size == 1)
  {
    for(Int i = 0; i < vn / 2; i++)
    {
      Vec re0 = vsrc0_re[i]; 
      Vec im0 = vsrc0_im[i]; 
      Vec re1 = vsrc1_re[i]; 
      Vec im1 = vsrc1_im[i]; 
      pair<Vec, Vec> dre = interleave<vec_size>(re0 + re1, re0 - re1);
      pair<Vec, Vec> dim = interleave<vec_size>(im0 + im1, im0 - im1);
      vdst_re[2 * i] = dre.first;
      vdst_re[2 * i + 1] = dre.second;
      vdst_im[2 * i] = dim.first;
      vdst_im[2 * i + 1] = dim.second;
    }    
  }
}

template<typename T>
void pass(
  Int n, Int dft_size,
  ComplexPtrs<T> src, ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  for(Int i = 0; i < n / 2; i += dft_size)
  {
    T* re0_ptr = src.re + i;
    T* im0_ptr = src.im + i;
    T* re1_ptr = src.re + n / 2 + i;
    T* im1_ptr = src.im + n / 2 + i;
    T* dst0_re_ptr = dst.re + 2 * i;
    T* dst0_im_ptr = dst.im + 2 * i;
    T* dst1_re_ptr = dst.re + 2 * i + dft_size;
    T* dst1_im_ptr = dst.im + 2 * i + dft_size;

    for(Int j = 0; j < dft_size; j++)
    {
      T tw_re = twiddle.re[j];
      T tw_im = twiddle.im[j];
      T re0 = re0_ptr[j];
      T im0 = im0_ptr[j];
      T re1 = re1_ptr[j];
      T im1 = im1_ptr[j];
      T mul_re = tw_re * re1 - tw_im * im1;
      T mul_im = tw_re * im1 + tw_im * re1;
      dst0_re_ptr[j] = re0 + mul_re;
      dst1_re_ptr[j] = re0 - mul_re;
      dst0_im_ptr[j] = im0 + mul_im;
      dst1_im_ptr[j] = im0 - mul_im;
    }
  }
}

void fft(
  Int n, ComplexPtrs<T> src, ComplexPtrs<T> twiddle, ComplexPtrs<T> dst)
{
  const Int vec_size = sizeof(Vec) / sizeof(T);
  bool result_in_dst = false;
  for(Int dft_size = 1; dft_size < n; dft_size *= 2)
  {
    auto tw = twiddle + (n - 2 * dft_size); 
    if(dft_size == 1)
      ct_dft_size_pass<1>(n, src, tw, dst);
    if(dft_size < vec_size)
      pass(n, dft_size, src, tw, dst);
    else
      pass(
        n / vec_size,
        dft_size / vec_size,
        (ComplexPtrs<Vec>&) src,
        (ComplexPtrs<Vec>&) tw,
        (ComplexPtrs<Vec>&) dst);

#if 0
    char buf[100];
    sprintf(buf, "src_re%d.float32", dft_size);
    dump(src.re, n, buf);
    sprintf(buf, "dst_re%d.float32", dft_size);
    dump(dst.re, n, buf);
#endif
    
    swap(src, dst);
    result_in_dst = !result_in_dst;
  }
  
  if(!result_in_dst)
    for(Int i = 0; i < n; i++)
    {
      dst.re[i] = src.re[i];
      dst.im[i] = src.im[i];
    }
}

void foo(Int n, const float* a, const float* b, float* c)
{
  auto va = (Vec*) a;
  auto vb = (Vec*) b;
  auto vc = (Vec*) c;
  auto vn = n * sizeof(T) / sizeof(Vec);
  for(Int i = 0; i < vn; i++)
  {
    auto r = interleave<4>(va[i], vb[i]);
    vc[2 * i] = r.first;
    vc[2 * i + 1] = r.second; 
  }
}

#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <fstream>

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

std::string tostr(Vec a)
{
  std::string r;
  for(Int i = 0; i < 8; i++)
  {
    if(i != 0) r += " ";
    r += std::to_string(((float*) &a)[i]);
  }

  return r;
}

template<typename T_>
void dump(T_* ptr, Int n, const char* name)
{
  std::ofstream(name, std::ios_base::binary).write((char*) ptr, sizeof(T_) * n);
}

int main()
{
  Int log2n = 10;
  Int n = 1 << log2n;
  ComplexPtrs<T> twiddle = {new T[n], new T[n]};
  init_twiddle(n, twiddle);

  ComplexPtrs<T> src = {new T[n], new T[n]};
  ComplexPtrs<T> dst = {new T[n], new T[n]};

  std::fill_n(dst.re, n, T(0));
  std::fill_n(dst.im, n, T(0));

  for(Int i = 0; i < n; i++)
  {
    src.im[i] = 0;
    T x = std::min(i, n - i) / T(n);
    src.re[i] = T(1) / (1 + 100 * x * x);
  }

  dump(twiddle.re, n, "twiddle_re.float32");
  dump(twiddle.im, n, "twiddle_im.float32");
  dump(src.re, n, "src_re.float32");
  dump(src.im, n, "src_im.float32");

  double t = get_time();
  for(int i = 0; i < 10000000000LL / (5 * n * log2n); i++)
    fft(n, src, twiddle, dst);

  printf("time %f\n", get_time() - t);

  dump(dst.re, n, "dst_re.float32");
  dump(dst.im, n, "dst_im.float32");

  return 0;
}
