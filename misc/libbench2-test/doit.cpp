/* this program is in the public domain */

#include "bench-user.h"
#include <math.h>
#include "fft.hpp"
#include <memory>
#include <vector>

BEGIN_BENCH_DOC
BENCH_DOC("name", "afft")
BENCH_DOC("version", "0.0")
BENCH_DOC("year", "2019")
BENCH_DOC("author", "Jernej Krempus")
BENCH_DOC("language", "C++")
END_BENCH_DOC

using Tr = afft::complex_transform<float, afft_interleaved>;
using ITr = afft::inverse_complex_transform<float, afft_interleaved>;
std::unique_ptr<Tr> tr;
std::unique_ptr<ITr> itr;

int can_do(bench_problem *p)
{
  return p->sz->rnk >= 1 && p->kind == PROBLEM_COMPLEX;
}

void setup(bench_problem *p)
{
  std::vector<size_t> dim;
  for(int i = 0; i < p->sz->rnk; i++)
    dim.push_back(size_t(p->sz->dims[i].n));

  if(p->sign == -1)
    tr = std::make_unique<Tr>(dim.size(), &dim[0]);
  else
    itr = std::make_unique<ITr>(dim.size(), &dim[0]);
}

void doit(int iter, bench_problem *p)
{
  int i;
  void *in = p->in;

  for(int i = 0; i < iter; i++)
  {
    if(p->sign == -1)
      (*tr)((float*) p->in, nullptr, (float*) p->out, nullptr);
    else
      (*itr)((float*) p->in, nullptr, (float*) p->out, nullptr);
  }
}

void done(bench_problem *p)
{
  if(p->sign == -1)
    tr.reset(nullptr);
  else
    itr.reset(nullptr);
}

void cleanup() { }
void main_init(int *argc, char ***argv) { }
