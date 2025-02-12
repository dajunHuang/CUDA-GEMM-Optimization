#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

#include "cutlass/gemm/device/gemm.h"

template <typename T>
void launch_gemm_kernel_07(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<T,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    T,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    T,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    typename CutlassGemm::Arguments args({m, n, k},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {*alpha, *beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
    
    gemm_operator(args);

    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_07<float>(size_t m, size_t n, size_t k,
                                             float const* alpha,
                                             float const* A, size_t lda,
                                             float const* B, size_t ldb,
                                             float const* beta, float* C,
                                             size_t ldc, cudaStream_t stream);
// Explicit instantiation.
template void launch_gemm_kernel_07<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);                                             