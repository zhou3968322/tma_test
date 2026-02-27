#pragma once

#include "tma_copy.h"

template <int TILE_M = 128, int TILE_N = 32, int THREADS = 32>
int copy_host_tma_swizzle_load_and_store_kernel(int M, int N,
                                                int iterations = 1) {
  using namespace cute;

  printf("Copy with TMA load and store -- swizzling enabled.\n");

  using Element = float;

  auto tensor_shape = make_shape(M, N);

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(float(i));

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  // Make tensors
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});

  // Reuse tma_copy.h kernel/params path, only switch SMEM layout to swizzle mode.
  // CUTLASS 4.2.1 TMA swizzle constraints require base exponent M in [4, 6].
  constexpr int kSwizzleB = 3;
  constexpr int kSwizzleM = 4;
  constexpr int kSwizzleS = 3;
  static_assert(4 <= kSwizzleM && kSwizzleM <= 6,
                "CUTLASS 4.2.1 expects TMA swizzle base exponent M in [4, 6].");
  using TmaSwizzle = Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>;

  // CUTLASS 4.2.1 runtime descriptor checks are sensitive to tile bytes on the
  // contiguous axis when swizzle is enabled. Keep TILE_N defaulted to 32 for
  // float so each row is 128B and descriptor initialization succeeds.
  static_assert(TILE_N * int(sizeof(Element)) == 128,
                "For CUTLASS 4.2.1 TMA swizzle path, default TILE_N should map to 128B rows.");

  auto smemBaseLayout = make_layout(tileShape, LayoutRight{});
  auto smemLayout = composition(TmaSwizzle{}, smemBaseLayout);

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout);

  Params params(tma_load, tma_store, gmemLayoutS, smemLayout, tileShape);

  dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
  dim3 blockDim(THREADS);

  int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
  printf("smem size: %d.\n", smem_size);

  void const *kernel =
      (void const *)copyTMAKernel<THREADS, Element, decltype(params)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 cluster_dims(1);

  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
    (void)status;
    cudaError result = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s)"
              << std::endl;
  }

  // Verify
  h_D = d_D;

  int good = 0, bad = 0;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_D[i] == h_S[i])
      good++;
    else
      bad++;
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;

  return (bad == 0) ? 0 : -1;
}
