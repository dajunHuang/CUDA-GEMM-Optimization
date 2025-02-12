#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main(int argc, char *argv[])
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{0U};

    double const fp64_abs_tol{1.0e-4f};
    double const fp64_rel_tol{0.0e-5f};

    size_t m{8192U};
    size_t n{8192U};
    size_t k{8192U};

    if(argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    size_t lda{(m + 16U - 1U) / 16U * 16U};
    size_t ldb{(k + 16U - 1U) / 16U * 16U};
    size_t ldc{(m + 16U - 1U) / 16U * 16U};

    std::cout << "Matrix Size: " << "M = " << m << " N = " << n << " K = " << k
              << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << ldc << std::endl;
    std::cout << std::endl;

    // Define all the GEMM kernel launch functions to be profiled.
    std::vector<std::pair<
        std::string,
        std::function<void(size_t, size_t, size_t, double const*, double const*,
                           size_t, double const*, size_t, double const*, double*,
                           size_t, cudaStream_t)>>> const
        gemm_kernel_launch_functions{
            // {"Custom GEMM Kernel 04", launch_gemm_kernel_04<double>},
            // {"Custom GEMM Kernel 07", launch_gemm_kernel_07<double>},
            {"Custom GEMM Kernel 08", launch_gemm_kernel_08<double>},
        };

    for (auto const& gemm_kernel_launch_function : gemm_kernel_launch_functions)
    {
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<double, double> const gemm_kernel_profile_result{
            profile_gemm<double>(
                m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                fp64_abs_tol, fp64_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return 0;
}