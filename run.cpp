#include "testing.hpp"

#include <iostream>

int main(int argc, char** argv)
{
  Options opt;

  auto result = parse_options(argc, argv, &opt);
  if(!result)
  {
    std::cerr << result.message << std::endl;
    return result.error ? 1 : 0;
  }

  return run_test(opt, std::cout) ? 0 : 1;
}
