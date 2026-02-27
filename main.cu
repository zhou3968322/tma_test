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
