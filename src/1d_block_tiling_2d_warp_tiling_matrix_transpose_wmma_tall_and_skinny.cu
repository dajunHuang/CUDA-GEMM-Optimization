#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"
#include <stdio.h>

// GEMM kernel v07 for tall and skinny MM.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WARP_TILE_SIZE_K, 
          size_t NUM_THREADS_PER_BLOCK>
__global__ void gemm_tall_and_skinny(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K]
                                               [BLOCK_TILE_SIZE_Y +
                                                BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X +
                                                        BLOCK_TILE_SKEW_SIZE_X];

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WARP_TILE_SIZE_Y,
                           WARP_TILE_SIZE_X, WARP_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major> a_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WARP_TILE_SIZE_Y,
                           WARP_TILE_SIZE_X, WARP_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major> b_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WARP_TILE_SIZE_Y,
                           WARP_TILE_SIZE_X, WARP_TILE_SIZE_K, T> acc_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WARP_TILE_SIZE_Y,
                           WARP_TILE_SIZE_X, WARP_TILE_SIZE_K, T> c_frag;

    // Make sure the accumulator starts from 0.
    nvcuda::wmma::fill_fragment(acc_frags, static_cast<T>(0));

    size_t const thread_linear_idx{threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

#pragma unroll
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        if(thread_linear_idx < BLOCK_TILE_SIZE_Y)
        {
            for(size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
            {
                A_thread_block_tile_transposed[k_i][thread_linear_idx] =
                    A[blockIdx.x * BLOCK_TILE_SIZE_Y + thread_linear_idx + 
                        (k_i + thread_block_tile_idx * BLOCK_TILE_SIZE_K) * lda];
            }
        }

        if(thread_linear_idx < BLOCK_TILE_SIZE_X)
        {
            for(size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
            {
                B_thread_block_tile[k_i][thread_linear_idx] = 
                    B[thread_block_tile_idx * BLOCK_TILE_SIZE_K + k_i + 
                        thread_linear_idx * ldb];
            }
        }

        __syncthreads();
        #pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K / WARP_TILE_SIZE_K; ++k_i)
        {
            nvcuda::wmma::load_matrix_sync(
                a_frags,
                & A_thread_block_tile_transposed[k_i * WARP_TILE_SIZE_K]
                                                [warp_row_idx * WARP_TILE_SIZE_Y],
                BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);

            nvcuda::wmma::load_matrix_sync(
                b_frags,
                &B_thread_block_tile[k_i * WARP_TILE_SIZE_K]
                                    [warp_col_idx * WARP_TILE_SIZE_X],
                BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);

            nvcuda::wmma::mma_sync(acc_frags, a_frags, b_frags, acc_frags);
        }
        __syncthreads();
    }


    // Load the fragment from global memory.
    nvcuda::wmma::load_matrix_sync(
        c_frag,
        &C[(blockIdx.x * BLOCK_TILE_SIZE_Y +
            warp_row_idx * WARP_TILE_SIZE_Y) +
            (warp_col_idx * WARP_TILE_SIZE_X) * ldc],
        ldc, nvcuda::wmma::mem_col_major);
    // Perform scaling and addition.
    for (size_t i{0}; i < c_frag.num_elements; ++i)
    {
        c_frag.x[i] = alpha * acc_frags.x[i] + beta * c_frag.x[i];
    }
    // Store the fragment back to global memory.
    nvcuda::wmma::store_matrix_sync(
        &C[(blockIdx.x * BLOCK_TILE_SIZE_Y +
            warp_row_idx * WARP_TILE_SIZE_Y)+
            (warp_col_idx * WARP_TILE_SIZE_X) * ldc],
        c_frag, ldc, nvcuda::wmma::mem_col_major);
}

template <typename T>
void launch_gemm_kernel_tall_and_skinny_01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    static_assert(BLOCK_TILE_SIZE_X == 32U);
    static_assert(BLOCK_TILE_SIZE_Y % BLOCK_TILE_SIZE_X == 0U);

    constexpr unsigned int WARP_TILE_SIZE_X{16U};
    constexpr unsigned int WARP_TILE_SIZE_Y{16U};
    constexpr unsigned int WARP_TILE_SIZE_K{16U};
    static_assert(BLOCK_TILE_SIZE_K % WARP_TILE_SIZE_K == 0U);
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y, 1U, 1U};
    gemm_tall_and_skinny<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
             WARP_TILE_SIZE_Y, WARP_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_tall_and_skinny_01<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);