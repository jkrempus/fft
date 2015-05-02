#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

typedef long Int;

template<typename T> T max(T a, T b){ return a > b ? a : b; }
template<typename T> T min(T a, T b){ return a < b ? a : b; }

Int to_strided_index(
  Int i, Int n, Int nchunks, Int chunk_size, Int dft_size, Int npasses,
  Int offset)
{
  if(chunk_size < dft_size)
  {
    Int ichunk = i / chunk_size;
    Int chunk_offset = i % chunk_size;
    Int dft_size_mul = 1 << npasses;

    return
      (ichunk & ~(dft_size_mul - 1)) * n / nchunks +
      (ichunk & (dft_size_mul - 1)) * dft_size +
      chunk_offset +
      dft_size_mul * (offset & ~(dft_size - 1)) +
      (offset & (dft_size - 1));
  }
  else
  {
    Int contiguous_size = chunk_size << npasses;

    Int icontiguous = i / contiguous_size;
    Int contiguous_offset = i % contiguous_size;

    Int stride = n / (nchunks * chunk_size / contiguous_size);

    return
      icontiguous * stride +
      contiguous_offset +
      offset / chunk_size * contiguous_size;
  }
}

void index_mappings(
  Int n,
  Int nchunks,
  Int chunk_size,
  Int dft_size,
  Int npasses,
  Int offset,
  std::vector<Int>& dst)
{
  dst.assign(npasses * nchunks * chunk_size, 0);
  for(Int i = 0; i < nchunks; i++)
    for(Int j = 0; j < chunk_size; j++)
      dst[i * chunk_size + j] = n / nchunks * i + j + offset;

  Int d = min(dft_size, chunk_size);

  for(Int pass = 1; pass < npasses; pass++)
  {
    Int l = nchunks * chunk_size / 2;
    Int* psrc = &dst[(pass - 1) * chunk_size * nchunks];
    Int* pdst = &dst[pass * chunk_size * nchunks];
    for(Int i = 0; i < l; i += d)
    {
      for(Int j = 0; j < d; j++)
      {
        Int isrc = i + j;
        Int idst0 = 2 * i + j;
        Int idst1 = 2 * i + j + d;
        Int mapped_isrc = psrc[isrc];
        Int multiple = mapped_isrc & ~(dft_size - 1);
        Int rem = mapped_isrc & (dft_size - 1);
        pdst[idst0] = 2 * multiple + rem;
        pdst[idst1] = 2 * multiple + rem + dft_size;
      }
    }

    dft_size *= 2;
    d *= 2;
  }
}

void pnm_vertical_stripes(FILE* f, bool* data, Int len)
{
  Int h = 100;
  fprintf(f, "P1\n%ld %ld\n", len, h);
  for(Int i = 0; i < h; i++)
    for(Int j = 0; j < len; j++)
      fprintf(f, "%d ", data[j]);
}

int main(int argc, const char** argv)
{
  Int n = 1 << 6;
  bool* a = new bool[n];
  std::fill_n(a, n, 0);

  Int nchunks = 16;
  Int chunk_size = 1;
  Int dft_size = 4;
  Int npasses = 5;
  Int offset = argc == 1 ? 0 : atoi(argv[1]);

  std::vector<Int> mappings;
  index_mappings(n, nchunks, chunk_size, dft_size, npasses, offset, mappings);
  for(Int i = 0; i < nchunks * chunk_size; i++)
  {
    for(Int j = 0; j < npasses; j++)
    {
      Int a = mappings[nchunks * chunk_size * j + i];
      Int b = to_strided_index(
        i, n, nchunks, chunk_size, dft_size, j, offset);

      printf("%ld %ld\t", a, b - a);
    }

    printf("\n");
  }

#if 0
  //for(Int offset = 0; offset < n / nchunks; offset += chunk_size)
  for(Int i = 0; i < nchunks * chunk_size; i++)
  {
    Int j = to_strided_index(i, n, nchunks, chunk_size, dft_size, npasses, offset);
    fprintf(stderr, "%ld %ld\n", i, j);
    a[j] = 1;
  }

  pnm_vertical_stripes(stdout, a, n);

#endif

  return 0;
}
