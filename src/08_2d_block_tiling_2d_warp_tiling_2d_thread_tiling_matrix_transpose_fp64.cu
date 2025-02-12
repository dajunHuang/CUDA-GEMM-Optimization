#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"
#include "stdio.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <mma.h>

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_A = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_B = 0U>
__device__ void load_data_from_global_memory_to_shared_memory_transposed(
    T const* A, size_t lda, T const* B, size_t ldb, T* A_thread_block_tile,
    T* B_thread_block_tile, size_t thread_block_tile_idx,
    size_t thread_linear_idx, size_t m, size_t n, size_t k)
{
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const A_row_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        // if (A_row_idx < m && A_col_idx < k)
        // {
        val = A[A_row_idx + A_col_idx * lda];
        // }
        // Removing the if will give another ~2 FLOPs performance on RTX
        // 3090. But it will make the kernel incorrect for some GEMM
        // configurations. T val{A[A_row_idx * lda + A_col_idx]}; This if
        // will slow down the kernel. Add static asserts from the host code
        // to guarantee this if is always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % NUM_THREADS ==
                      0U);

        constexpr size_t ldsa = BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_A;
        A_thread_block_tile[A_thread_block_tile_row_idx +
                            A_thread_block_tile_col_idx * ldsa] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        // if (B_row_idx < k && B_col_idx < n)
        // {
        val = B[B_row_idx + B_col_idx * ldb];
        // }
        // Removing the if will give another ~2 FLOPs performance on RTX
        // 3090. But it will make the kernel incorrect for some GEMM
        // configurations. T val{B[B_row_idx * ldb + B_col_idx]}; This if
        // will slow down the kernel. Add static asserts from the host code
        // to guarantee this if is always true.
        static_assert(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);

        constexpr size_t ldsb = BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_B;
        B_thread_block_tile[B_thread_block_tile_row_idx * ldsb +
                            B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_A,
          size_t BLOCK_TILE_SKEW_SIZE_B, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_08(size_t m, size_t n, size_t k, T const alpha, T const* A,
                        size_t lda, T const* B, size_t ldb, T const beta, T* C,
                        size_t ldc)
{
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    // Cache a tile of A and B in shared memory for data reuse.
    // A is col majored, B is row majored, C is col majored.
    constexpr size_t ldsa = BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_A;
    __shared__ alignas(16) T A_thread_block_tile[BLOCK_TILE_SIZE_K * ldsa];
    constexpr size_t ldsb = BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_B;
    __shared__ alignas(16) T B_thread_block_tile[BLOCK_TILE_SIZE_K * ldsb];

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_X,
                           WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_X,
                           WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_X,
                           WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_X][NUM_WMMA_TILES_Y];

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_X;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_Y;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx{threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_Y};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_Y};

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_from_global_memory_to_shared_memory_transposed<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS, BLOCK_TILE_SKEW_SIZE_A, BLOCK_TILE_SKEW_SIZE_B>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
            
            #pragma unroll
            for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
            {
                __syncthreads();
#pragma unroll
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_X; ++wmma_tile_row_idx)
            {
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_tile_row_idx],
                    &A_thread_block_tile[warp_row_idx * WARP_TILE_SIZE_X +
                                         wmma_tile_row_idx * WMMA_TILE_SIZE_X +
                                         k_i * WMMA_TILE_SIZE_K * ldsa],
                    ldsa);
            }
#pragma unroll
            for (size_t wmma_tile_col_idx{0U};
                 wmma_tile_col_idx < NUM_WMMA_TILES_Y; ++wmma_tile_col_idx)
            {
                nvcuda::wmma::load_matrix_sync(
                    b_frags[wmma_tile_col_idx],
                    &B_thread_block_tile[warp_col_idx * WARP_TILE_SIZE_Y +
                                         wmma_tile_col_idx * WMMA_TILE_SIZE_Y +
                                         k_i * WMMA_TILE_SIZE_K * ldsb],
                    ldsb);
            }
#pragma unroll
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_X; ++wmma_tile_row_idx)
            {
#pragma unroll
                for (size_t wmma_tile_col_idx{0U};
                     wmma_tile_col_idx < NUM_WMMA_TILES_Y; ++wmma_tile_col_idx)
                {
                    // Perform the matrix multiplication.
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                        a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
    }

    // Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_X;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_Y;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::store_matrix_sync(
                &C[blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_row_idx * WARP_TILE_SIZE_X +
                   wmma_tile_row_idx * WMMA_TILE_SIZE_X +
                   (blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_col_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_col_idx * WMMA_TILE_SIZE_Y) *
                       ldc],
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx], ldc,
                nvcuda::wmma::mem_col_major);
        }
    }
}

template <typename T>
void launch_gemm_kernel_08(size_t m, size_t n, size_t k, T const* alpha,
                           T const* A, size_t lda, T const* B, size_t ldb,
                           T const* beta, T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{32U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_A{0U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_B{1U};

    constexpr unsigned int WMMA_TILE_SIZE_X{8U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{8U};
    constexpr unsigned int WMMA_TILE_SIZE_K{4U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_08<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            BLOCK_TILE_SKEW_SIZE_A, BLOCK_TILE_SKEW_SIZE_B, WARP_TILE_SIZE_X,
            WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y,
            WMMA_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_gemm_kernel_08<double>(size_t m, size_t n, size_t k,
                                            double const* alpha,
                                            double const* A, size_t lda,
                                            double const* B, size_t ldb,
                                            double const* beta, double* C,
                                            size_t ldc, cudaStream_t stream);
