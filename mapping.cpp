#include <cstdio>
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
    Int ifinal_dft = ichunk / dft_size_mul;
    Int final_dft_offset = ichunk % dft_size_mul;

    Int final_dft_stride = n / (nchunks / dft_size_mul);

    Int offset0 = offset / dft_size;
    Int offset1 = offset % dft_size;

    return
      ifinal_dft * final_dft_stride +
      final_dft_offset * dft_size +
      (dft_size_mul * dft_size) * offset0 +
      chunk_offset + offset1;
  }
  else
  {
    Int dft_size_mul = 1 << npasses;
    Int current_dft_size = dft_size << npasses;
    Int contiguous_size = max(dft_size, chunk_size) << npasses;

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

int main()
{
  Int n = 1 << 11;
  bool* a = new bool[n];
  std::fill_n(a, n, 0);

  Int nchunks = 16;
  Int chunk_size = 4;
  Int dft_size = 16;
  Int npasses = 5;
  Int offset = 4;

  std::vector<Int> mappings;
  index_mappings(n, nchunks, chunk_size, dft_size, npasses, offset, mappings);
  for(Int i = 0; i < nchunks * chunk_size; i++)
  {
    for(Int j = 0; j < npasses; j++)
    {
      printf("%ld\t", mappings[nchunks * chunk_size * j + i]);
      printf("%ld\t", to_strided_index(
        i, n, nchunks, chunk_size, dft_size, j, offset));
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
