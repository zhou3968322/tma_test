#include <cstring>

#include "cutlass/util/command_line.h"

#include "tma_copy_swizzle.h"

int main(int argc, char const **argv)
{

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, iterations, swizzle;
  cmd.get_cmd_line_argument("M", M, 1024);
  cmd.get_cmd_line_argument("N", N, 1024);
  cmd.get_cmd_line_argument("iterations", iterations, 1);
  cmd.get_cmd_line_argument("swizzle", swizzle, 1);

  // Support additional CLI forms:
  //   ./main swizzle
  //   ./main noswizzle
  //   ./main --swizzle 0|1
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "swizzle") == 0 ||
        std::strcmp(argv[i], "--swizzle") == 0) {
      if ((i + 1) < argc &&
          (std::strcmp(argv[i + 1], "0") == 0 ||
           std::strcmp(argv[i + 1], "1") == 0)) {
        swizzle = (std::strcmp(argv[i + 1], "1") == 0) ? 1 : 0;
      } else {
        swizzle = 1;
      }
    } else if (std::strcmp(argv[i], "noswizzle") == 0 ||
               std::strcmp(argv[i], "--noswizzle") == 0) {
      swizzle = 0;
    }
  }

  std::cout << "(M, N): " << M << ", " << N << std::endl;
  std::cout << "swizzle: " << swizzle << std::endl;

  if (swizzle) {
    copy_host_tma_swizzle_load_and_store_kernel(M, N, iterations);
  } else {
    copy_host_tma_load_and_store_kernel(M, N, iterations);
  }

  // scaleTmaKernelHost(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 2>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 2>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 4>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 4>(M, N, iterations);


  return 0;
}
