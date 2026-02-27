#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool is_elected() {
  unsigned int tid = threadIdx.x;
  unsigned int warp_id = tid / 32;
  unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
  return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
}

__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map, int x,
                           int y) {
  __shared__ alignas(128) half smem_buffer[32 * 2];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  barrier::arrival_token token;
  if (is_elected()) {
    int32_t tensor_coords[2] = {x, y};
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              &smem_buffer, &tensor_map, tensor_coords,
                              cuda::device::barrier_native_handle(bar));

    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    token = bar.arrive();
  }

  bar.wait(std::move(token));
  for (int i = 0; i < 1000; ++i) {
    if (threadIdx.x < 32) {
      smem_buffer[threadIdx.x] = __hadd(
          smem_buffer[threadIdx.x],
          __float2half(__half2float(smem_buffer[threadIdx.x + 32])));
    }
  }
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  if (is_elected()) {
    int32_t tensor_coords[2] = {x, y};
    ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared,
                              &tensor_map, tensor_coords, &smem_buffer);

    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }

  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}

int main() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  std::cout << "GPU: " << prop.name << ", Compute Capability: " << prop.major
            << "." << prop.minor << std::endl;

  if (prop.major < 9) {
    std::cerr
        << "ERROR: TMA requires Compute Capability 9.0 (Hopper). Your GPU is "
           "not supported!"
        << std::endl;
    return -1;
  }

  constexpr int M = 128;
  constexpr int K = 257;
  constexpr int K_align = 320;
  constexpr size_t data_size = M * K_align * sizeof(half);

  std::vector<half> h_data(M * K_align);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K_align; ++j) {
      if (j < K) {
        h_data[i * K_align + j] = __float2half(static_cast<float>(i * K_align + j));
      } else {
        h_data[i * K_align + j] = __float2half(-1.0f);
      }
    }
  }

  half *d_data = nullptr;
  cudaMalloc(&d_data, data_size);
  cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);

  void *tensor_ptr = d_data;

  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {K, M};
  uint64_t stride[rank - 1] = {K_align * sizeof(half)};
  uint32_t box_size[rank] = {32, 2};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank,
      tensor_ptr, size, stride, box_size, elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to create tensor map: " << res << std::endl;
    return -1;
  }

  const int WARMUP_RUNS = 10;
  const int TIMED_RUNS = 1000;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Starting warm-up (" << WARMUP_RUNS << " runs)..." << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    kernel_tma<<<1, 32>>>(tensor_map, 0, 0);
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Warm-up failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  std::cout << "Starting timing (" << TIMED_RUNS << " runs)..." << std::endl;

  cudaEventRecord(start, 0);

  for (int i = 0; i < TIMED_RUNS; ++i) {
    kernel_tma<<<1, 32>>>(tensor_map, 0, 0);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
              << std::endl;
  } else {
    float total_time_ms = 0.0f;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    float avg_time_ms = total_time_ms / TIMED_RUNS;
    float bytes_per_kernel = sizeof(half) * 32 * 2 * 2;
    float total_bytes = bytes_per_kernel * TIMED_RUNS;
    float effective_bandwidth_gbps =
        (total_bytes / (total_time_ms / 1000.0f)) / 1e9f;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total time for " << TIMED_RUNS << " runs: " << total_time_ms
              << " ms" << std::endl;
    std::cout << "Average time per kernel: " << avg_time_ms << " ms"
              << std::endl;
    std::cout << "Effective Bandwidth (SMEM <-> Global): "
              << effective_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_data);

  return 0;
}
