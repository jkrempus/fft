#include <cstdio>
#include <algorithm>

typedef long Int;

Int to_strided_index(
  Int i, Int n, Int nchunks, Int chunk_size, Int dft_size, Int npasses,
  Int offset)
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
    (mul * dft_size) * offset0 +
    chunk_offset + offset1;
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
  Int n = 1024;
  bool* a = new bool[n];
  std::fill_n(a, n, 0);

  Int nchunks = 16;
  Int chunk_size = 4;
  Int dft_size = 16;
  Int npasses = 4;

  Int offset = 0;
  //for(Int offset = 0; offset < n / nchunks; offset += chunk_size)
  for(Int i = 0; i < nchunks * chunk_size; i++)
  {
    Int j = to_strided_index(i, n, nchunks, chunk_size, dft_size, npasses, offset);
    fprintf(stderr, "%ld %ld\n", i, j);
    a[j] = 1;
  }

  pnm_vertical_stripes(stdout, a, n);

  return 0;
}
