#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"
#include <cooperative_groups.h>
#include "stdio.h"
#include <cstddef>
#include <cstdio>
#include <iostream>

// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t NUM_THREADS_PER_BLOCK>
// __device__ void set_global_memory_to_zero(T * C_thread_block_tile,
//     size_t ldsc, size_t warp_linear_idx, size_t thread_linear_idx_in_warp)
// {
//     constexpr unsigned int NUM_EACH_THREAD_LOAD_C{(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) /
//         NUM_THREADS_PER_BLOCK};
//     constexpr unsigned int NUM_LOAD_THREADS_PER_ROW_C{BLOCK_TILE_SIZE_Y /  
//         NUM_EACH_THREAD_LOAD_C};
//     size_t const c_load_warp_row_idx{warp_linear_idx / NUM_LOAD_THREADS_PER_ROW_C};
//     size_t const c_load_warp_col_idx{warp_linear_idx % NUM_LOAD_THREADS_PER_ROW_C};

//     size_t const block_row{c_load_warp_row_idx * 32U + thread_linear_idx_in_warp};
//     size_t const block_col{c_load_warp_col_idx * NUM_EACH_THREAD_LOAD_C};

//     for(size_t load_time{0}; load_time < NUM_EACH_THREAD_LOAD_C; ++load_time)
//     {
//         C_thread_block_tile[block_row + (block_col + load_time) * ldsc] = 0;
//     }
// }

template <typename T, size_t BLOCK_TILE_SIZE, size_t WARP_TILE_SIZE,
          size_t NUM_THREAD_TILES_PER_WARP, size_t THREAD_TILE_SIZE>
__device__ void load_data_from_shared_memory_to_register_file(
    T const *thread_block_tile,
    T register_values[NUM_THREAD_TILES_PER_WARP][THREAD_TILE_SIZE],
    size_t warp_idx, size_t thread_idx)
{
    static_assert(BLOCK_TILE_SIZE % THREAD_TILE_SIZE == 0U);
#pragma unroll
    for (size_t thread_tile_repeat_idx{0U};
         thread_tile_repeat_idx < NUM_THREAD_TILES_PER_WARP;
         ++thread_tile_repeat_idx)
    {
        size_t const thread_block_tile_idx{
            warp_idx * WARP_TILE_SIZE + thread_tile_repeat_idx *
                (WARP_TILE_SIZE / NUM_THREAD_TILES_PER_WARP) +
                thread_idx * THREAD_TILE_SIZE};
#pragma unroll
        for (size_t thread_tile_idx{0U}; thread_tile_idx < THREAD_TILE_SIZE;
                ++thread_tile_idx)
        {
            register_values[thread_tile_repeat_idx][thread_tile_idx] = 
                thread_block_tile[thread_block_tile_idx + thread_tile_idx];
        }
    }
}

template <typename T, size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
          size_t NUM_THREAD_TILES_PER_WARP_X, size_t NUM_THREAD_TILES_PER_WARP_Y,
          size_t NUM_THREADS_PER_WARP_X, size_t NUM_THREADS_PER_WARP_Y,
          size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y>
__device__ void compute_thread_tile_results(
    T const A_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X],
    T const B_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y],
    T C_thread_results[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y],
    size_t const thread_linear_row_idx_in_warp,
    size_t const thread_linear_col_idx_in_warp)
{
#pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U};
         thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_X;
         ++thread_tile_repeat_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U};
             thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_Y;
             ++thread_tile_repeat_col_idx)
        {
            size_t const C_thread_row_idx{
                thread_tile_repeat_row_idx * (WARP_TILE_SIZE_X /
                NUM_THREAD_TILES_PER_WARP_X) +
                thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_X};
            size_t const C_thread_col_idx{
                thread_tile_repeat_col_idx * (WARP_TILE_SIZE_Y /
                NUM_THREAD_TILES_PER_WARP_Y) +
                thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_Y};
#pragma unroll
            for (size_t thread_tile_x_idx{0U};
                 thread_tile_x_idx < THREAD_TILE_SIZE_X; ++thread_tile_x_idx)
            {
#pragma unroll
                for (size_t thread_tile_y_idx{0U};
                     thread_tile_y_idx < THREAD_TILE_SIZE_Y;
                     ++thread_tile_y_idx)
                {
                    // TODO transpose? Is there register bank?
                    // What's the best order for loops?
                    // Right?
                    C_thread_results[C_thread_row_idx + thread_tile_x_idx]
                        [C_thread_col_idx + thread_tile_y_idx] +=
                        A_vals[thread_tile_repeat_row_idx][thread_tile_x_idx] *
                        B_vals[thread_tile_repeat_col_idx][thread_tile_y_idx];
                }
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t NUM_WARPS_X, size_t NUM_WARPS_Y,
          size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
          size_t NUM_THREAD_TILES_PER_WARP_X, size_t NUM_THREAD_TILES_PER_WARP_Y,
          size_t NUM_THREADS_PER_WARP_X, size_t NUM_THREADS_PER_WARP_Y,
          size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y>
__device__ void copy_results_from_register_to_shared_memory(
    T C_thread_results[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y],
    T *C_thread_block_tile, size_t const ldc, size_t const m, size_t const n,
    size_t const warp_row_idx, size_t const warp_col_idx,
    size_t const thread_linear_row_idx_in_warp,
    size_t const thread_linear_col_idx_in_warp)
{
#pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U};
         thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_X;
         ++thread_tile_repeat_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U};
             thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_Y;
             ++thread_tile_repeat_col_idx)
        {
            size_t const C_thread_warp_row_idx{
                thread_tile_repeat_row_idx * (WARP_TILE_SIZE_X /
                NUM_THREAD_TILES_PER_WARP_X) +
                thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_X};
            size_t const C_thread_warp_col_idx{
                thread_tile_repeat_col_idx * (WARP_TILE_SIZE_Y /
                NUM_THREAD_TILES_PER_WARP_Y) +
                thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_Y};
            size_t const C_thread_block_tile_row_idx{
                warp_row_idx * WARP_TILE_SIZE_X + C_thread_warp_row_idx};
            size_t const C_thread_block_tile_col_idx{
                warp_col_idx * WARP_TILE_SIZE_Y + C_thread_warp_col_idx};
#pragma unroll
            for (size_t thread_tile_x_idx{0U};
                 thread_tile_x_idx < THREAD_TILE_SIZE_X; ++thread_tile_x_idx)
            {
#pragma unroll
                for (size_t thread_tile_y_idx{0U};
                     thread_tile_y_idx < THREAD_TILE_SIZE_Y;
                     ++thread_tile_y_idx)
                {
                    C_thread_block_tile[(C_thread_block_tile_row_idx + thread_tile_x_idx) +
                        (C_thread_block_tile_col_idx + thread_tile_y_idx) * ldc] = 
                        C_thread_results[C_thread_warp_row_idx + thread_tile_x_idx]
                        [C_thread_warp_col_idx + thread_tile_y_idx];

                }
            }
        }
    }
}

// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t NUM_THREADS_PER_BLOCK>
// __device__ void copy_results_from_shared_to_global_memory(
//     size_t block_row_idx, size_t block_col_idx, T const* C_thread_block_tile,
//     size_t ldsc, T* C_work, size_t ldc, size_t warp_linear_idx,
//     size_t thread_linear_idx_in_warp)
// {
//     constexpr unsigned int NUM_EACH_THREAD_LOAD_C{(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) /
//         NUM_THREADS_PER_BLOCK};
//     constexpr unsigned int NUM_LOAD_THREADS_PER_ROW_C{BLOCK_TILE_SIZE_Y /  
//         NUM_EACH_THREAD_LOAD_C};
//     size_t const c_load_warp_row_idx{warp_linear_idx / NUM_LOAD_THREADS_PER_ROW_C};
//     size_t const c_load_warp_col_idx{warp_linear_idx % NUM_LOAD_THREADS_PER_ROW_C};

//     size_t const block_row{c_load_warp_row_idx * 32U + thread_linear_idx_in_warp};
//     size_t const block_col{c_load_warp_col_idx * NUM_EACH_THREAD_LOAD_C};
//     size_t const c_row{block_row_idx * BLOCK_TILE_SIZE_X + block_row};
//     size_t const c_col{block_col_idx * BLOCK_TILE_SIZE_Y + block_col};

//     for(size_t load_time{0}; load_time < NUM_EACH_THREAD_LOAD_C; ++load_time)
//     {
//         C_work[c_row + (c_col + load_time) * ldc] +=
//             C_thread_block_tile[block_row + (block_col + load_time) * ldsc];
//     }
// }

template <typename T, size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SKEW_SIZE_A, size_t BLOCK_TILE_SKEW_SIZE_B,
          size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
          size_t NUM_THREADS_PER_WARP_X, size_t NUM_THREADS_PER_WARP_Y,
          size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_tall_and_skinny(size_t m, size_t n, size_t k, T const* alpha,
                                     T const* A, size_t lda, T const* B, size_t ldb,
                                     T const* beta, T* C, size_t ldc)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y * 32U};

    // Cache a tile of A and B in shared memory for data reuse.
    // A is col majored, B is row majored, C is col majored.
    constexpr size_t ldsa = BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_A;
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_K * ldsa];
    constexpr size_t ldsb = BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_B;
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K * ldsb];
    constexpr size_t ldsc = BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_A;
    // TODO shared memory of C should be initialized to all 0.
    __shared__ T C_thread_block_tile[BLOCK_TILE_SIZE_Y * ldsc];

    size_t const thread_linear_idx{threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};

    // set_global_memory_to_zero<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
    //     NUM_THREADS_PER_BLOCK>
    //     (C_thread_block_tile, ldsc, warp_linear_idx, thread_linear_idx_in_warp);


    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
        static_cast<T>(0)};

    constexpr unsigned int NUM_EACH_THREAD_LOAD_A{
        (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) / NUM_THREADS_PER_BLOCK};
    constexpr unsigned int NUM_LOAD_THREADS_PER_ROW_A{
        BLOCK_TILE_SIZE_K / NUM_EACH_THREAD_LOAD_A};
    size_t const a_load_warp_row_idx{warp_linear_idx / NUM_LOAD_THREADS_PER_ROW_A};
    size_t const a_load_warp_col_idx{warp_linear_idx % NUM_LOAD_THREADS_PER_ROW_A};

    // if(blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("threadIdx: %ld a_load_warp_row_idx: %ld a_load_warp_col_idx: %ld NUM_EACH_THREAD_LOAD_A: %d NUM_LOAD_THREADS_PER_ROW_A: %d\n",
    //         thread_linear_idx, a_load_warp_row_idx, a_load_warp_col_idx, NUM_EACH_THREAD_LOAD_A, NUM_LOAD_THREADS_PER_ROW_A);
    // }

    #pragma unroll
    for(size_t load_time{0}; load_time < NUM_EACH_THREAD_LOAD_A; ++load_time)
    {
        A_thread_block_tile[(a_load_warp_row_idx * 32U + thread_linear_idx_in_warp) +
                            (a_load_warp_col_idx * NUM_EACH_THREAD_LOAD_A + load_time) * ldsa] =
            A[(blockIdx.x * BLOCK_TILE_SIZE_X + a_load_warp_row_idx * 32U + thread_linear_idx_in_warp) +
              (blockIdx.y * BLOCK_TILE_SIZE_K + a_load_warp_col_idx * NUM_EACH_THREAD_LOAD_A + load_time) * lda];
    }

    constexpr unsigned int NUM_EACH_THREAD_LOAD_B{(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y) / NUM_THREADS_PER_BLOCK};
    constexpr unsigned int NUM_LOAD_THREADS_PER_ROW_B{BLOCK_TILE_SIZE_Y / NUM_EACH_THREAD_LOAD_B};
    size_t const b_load_warp_row_idx{warp_linear_idx / NUM_LOAD_THREADS_PER_ROW_B};
    size_t const b_load_warp_col_idx{warp_linear_idx % NUM_LOAD_THREADS_PER_ROW_B};

    // if(blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("threadIdx: %ld b_load_warp_row_idx: %ld b_load_warp_col_idx: %ld NUM_EACH_THREAD_LOAD_B: %d NUM_LOAD_THREADS_PER_ROW_B: %d\n",
    //         thread_linear_idx, b_load_warp_row_idx, b_load_warp_col_idx, NUM_EACH_THREAD_LOAD_B, NUM_LOAD_THREADS_PER_ROW_B);
    // }

    #pragma unroll
    for(size_t load_time{0}; load_time < NUM_EACH_THREAD_LOAD_B; ++load_time)
    {
        B_thread_block_tile[(b_load_warp_row_idx * 32U + thread_linear_idx_in_warp) * ldsb +
                            (b_load_warp_col_idx * NUM_EACH_THREAD_LOAD_B + load_time)] =
            B[(blockIdx.y * BLOCK_TILE_SIZE_K + b_load_warp_row_idx * 32U + thread_linear_idx_in_warp) +
              (b_load_warp_col_idx * NUM_EACH_THREAD_LOAD_B + load_time) * ldb];
    }
    __syncthreads();

    // Check shared memory load
    //
    // if(thread_linear_idx == 0)
    // {
    //     // printf("BLOCK_TILE_SIZE_X: %ld BLOCK_TILE_SIZE_K: %ld\n", BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_K);
    //     for(size_t x_i{0}; x_i < BLOCK_TILE_SIZE_X; ++x_i)
    //     {
    //         for(size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    //         {
    //             if(A[(blockIdx.x * BLOCK_TILE_SIZE_X + x_i) +
    //             (blockIdx.y * BLOCK_TILE_SIZE_K + k_i) * lda]
    //             != A_thread_block_tile[x_i + k_i * ldsa])
    //                 printf("A share meme copy err, A_thread_block_tile[%ld][%ld]\n", x_i, k_i);
    //         }
    //     }
    //     for(size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    //     {
    //         for(size_t y_i{0}; y_i < BLOCK_TILE_SIZE_Y; ++y_i)
    //         {
    //             if(B[(blockIdx.y * BLOCK_TILE_SIZE_K + k_i) + (y_i) * ldb] !=
    //                B_thread_block_tile[k_i * ldsb + y_i])
    //                 printf("A share meme copy err, B_thread_block_tile[%ld][%ld]\n", k_i, y_i);
    //         }
    //     }
    // }

    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_Y};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_Y};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
        NUM_THREADS_PER_WARP_Y};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
        NUM_THREADS_PER_WARP_Y};

    T C_thread_results[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y] = {
        static_cast<T>(0)};

    #pragma unroll
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    {
        // Load data from shared memory to register file for A.
        load_data_from_shared_memory_to_register_file<T, BLOCK_TILE_SIZE_X,
            WARP_TILE_SIZE_X, NUM_THREAD_TILES_PER_WARP_X, 
            THREAD_TILE_SIZE_X>
            (A_thread_block_tile + k_i * ldsa, A_vals, warp_row_idx, thread_linear_row_idx_in_warp);

        // Load data from shared memory to register file for B.
        load_data_from_shared_memory_to_register_file<T, BLOCK_TILE_SIZE_Y,
            WARP_TILE_SIZE_Y, NUM_THREAD_TILES_PER_WARP_Y,
            THREAD_TILE_SIZE_Y>
            (B_thread_block_tile + k_i * ldsb, B_vals, warp_col_idx, thread_linear_col_idx_in_warp);

        // Check register load
        //
        // for (size_t thread_tile_repeat_idx{0U};
        //     thread_tile_repeat_idx < NUM_THREAD_TILES_PER_WARP_X;
        //     ++thread_tile_repeat_idx)
        // {
        //     size_t const thread_block_tile_idx{
        //         warp_row_idx * WARP_TILE_SIZE_X + thread_tile_repeat_idx *
        //             (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
        //             thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_X};
        //     for (size_t thread_tile_idx{0U}; thread_tile_idx < THREAD_TILE_SIZE_X;
        //             ++thread_tile_idx)
        //     {
        //         if(A_vals[thread_tile_repeat_idx][thread_tile_idx] != 
        //             A_thread_block_tile[k_i * ldsa + thread_block_tile_idx + thread_tile_idx])
        //         {
        //             printf("A reg load error\n");   
        //         }

        //     }
        // }
        // for (size_t thread_tile_repeat_idx{0U};
        //     thread_tile_repeat_idx < NUM_THREAD_TILES_PER_WARP_Y;
        //     ++thread_tile_repeat_idx)
        // {
        //     size_t const thread_block_tile_idx{
        //         warp_col_idx * WARP_TILE_SIZE_Y + thread_tile_repeat_idx *
        //             (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
        //             thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_Y};
        //     for (size_t thread_tile_idx{0U}; thread_tile_idx < THREAD_TILE_SIZE_Y;
        //             ++thread_tile_idx)
        //     {
        //         if(B_vals[thread_tile_repeat_idx][thread_tile_idx] != 
        //             B_thread_block_tile[k_i * ldsb + thread_block_tile_idx + thread_tile_idx])
        //         {
        //             printf("B reg load error\n");   
        //         }

        //     }
        // }

        // Compute NUM_THREAD_TILES_PER_WARP_X * NUM_THREAD_TILES_PER_WARP_Y
        // outer products.
        compute_thread_tile_results<T, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                                    NUM_THREAD_TILES_PER_WARP_X,
                                    NUM_THREAD_TILES_PER_WARP_Y,
                                    NUM_THREADS_PER_WARP_X,
                                    NUM_THREADS_PER_WARP_Y,
                                    THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
                                    (A_vals, B_vals, C_thread_results,
                                        thread_linear_row_idx_in_warp,
                                        thread_linear_col_idx_in_warp);   
    }

    __syncthreads();

    // Add the results of threads to shared_memory.
    copy_results_from_register_to_shared_memory<T,
        BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
        NUM_WARPS_X, NUM_WARPS_Y,
        WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
        NUM_THREAD_TILES_PER_WARP_X, NUM_THREAD_TILES_PER_WARP_Y,
        NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y,
        THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
        (C_thread_results, C_thread_block_tile, ldsc, m, n, warp_row_idx, warp_col_idx,
        thread_linear_row_idx_in_warp, thread_linear_col_idx_in_warp);

    // Check register add result
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //     for(size_t x_i{0}; x_i < BLOCK_TILE_SIZE_X; ++x_i)
    //     {
    //         for(size_t y_i{0}; y_i < BLOCK_TILE_SIZE_Y; ++y_i)
    //         {
    //             T sum{0};
    //             for(size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
    //             {
    //                 sum += A[(blockIdx.x * BLOCK_TILE_SIZE_X + x_i) +
    //                     (blockIdx.y * BLOCK_TILE_SIZE_K + k_i) * lda] *
    //                     B[(blockIdx.y * BLOCK_TILE_SIZE_K + k_i) +
    //                     (y_i) * ldb];
    //             }
    //             if(sum != C_thread_block_tile[x_i + y_i * ldsc])
    //             {
    //                 printf("C reg add error x_i: %ld y_i: %ld sum: %f c_block_tile: %f\n", x_i, y_i, sum, C_thread_block_tile[x_i + y_i * ldsc]);
    //             }
    //         }
    //     }
    // }

    // Copy the results of blocks to global memeory.
    // copy_results_from_shared_to_global_memory<T, BLOCK_TILE_SIZE_X,
    //     BLOCK_TILE_SIZE_Y, NUM_THREADS_PER_BLOCK>
    //     (blockIdx.x, blockIdx.y,
    //     C_thread_block_tile, ldsc, C_work, ldc,
    //     warp_linear_idx, thread_linear_idx_in_warp);

}

template <typename T>
void launch_gemm_kernel_tall_and_skinny_03(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};  // == n

    if(m % BLOCK_TILE_SIZE_X != 0 || n % BLOCK_TILE_SIZE_Y != 0 ||
        k % BLOCK_TILE_SIZE_K != 0)
    {
        printf("Block constraint not satisfied\n");
        return;
    }

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{32U};   // == n

    constexpr unsigned int NUM_THREADS_PER_WARP_X{8U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{4U};
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);

    constexpr unsigned int THREAD_TILE_SIZE_X{4U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};

    static_assert(WARP_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(WARP_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_A{0U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_B{1U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y * 32U};

    static_assert(BLOCK_TILE_SIZE_X % 32U == 0);
    static_assert(((BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) % NUM_THREADS_PER_BLOCK) == 0);
    constexpr unsigned int NUM_EACH_THREAD_LOAD_A{
        (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) / NUM_THREADS_PER_BLOCK};
    static_assert((BLOCK_TILE_SIZE_K) % (NUM_EACH_THREAD_LOAD_A) == 0);

    static_assert(BLOCK_TILE_SIZE_K % 32U == 0);
    static_assert(((BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y) % NUM_THREADS_PER_BLOCK) == 0);
    constexpr unsigned int NUM_EACH_THREAD_LOAD_B{
        (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y) / NUM_THREADS_PER_BLOCK};
    static_assert((BLOCK_TILE_SIZE_Y) % (NUM_EACH_THREAD_LOAD_B) == 0);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(k) + BLOCK_TILE_SIZE_K - 1U) /
            BLOCK_TILE_SIZE_K,
        1U};

    // printf("Block Dim: %d Grid Dim.x: %d Grid Dim.y: %d\n", block_dim.x, grid_dim.x, grid_dim.y);

    gemm_tall_and_skinny<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_Y,
                         BLOCK_TILE_SKEW_SIZE_A, BLOCK_TILE_SKEW_SIZE_B,
                         WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                         NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y,
                         THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
                         <<<grid_dim, block_dim, 0U, stream>>>
                         (m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Explicit instantiation.
template void launch_gemm_kernel_tall_and_skinny_03<float>(size_t m, size_t n, size_t k,
                                                           float const* alpha,
                                                           float const* A, size_t lda,
                                                           float const* B, size_t ldb,
                                                           float const* beta, float* C,
                                                           size_t ldc, cudaStream_t stream);
