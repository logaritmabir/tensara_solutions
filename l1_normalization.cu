#include <cuda_runtime.h>

__device__ __forceinline__ size_t get_linear_index(size_t x, size_t y, size_t width){
    return x * width + y;
}

__global__ void l1_normalization(const float* __restrict__ X, float* __restrict__ Y, size_t cols){
    extern __shared__ float s_sum[];
    
    size_t row_idx = blockIdx.x;
    size_t col_idx = threadIdx.x;

    float local_sum = 0.0f;
    for(size_t i = col_idx; i < cols; i += blockDim.x){
        local_sum += fabsf(X[get_linear_index(row_idx, i, cols)]);
    }
    s_sum[col_idx] = local_sum;
    __syncthreads();

    for(size_t s = blockDim.x/2; s > 0 ; s>>=1){
        if(col_idx < s){
            s_sum[col_idx] += s_sum[col_idx + s];
        }
        __syncthreads();
    }

    float normalized = s_sum[0];

    for(size_t i = col_idx; i < cols; i += blockDim.x){
        size_t pos = get_linear_index(row_idx, i, cols);
        Y[pos] = X[pos] / normalized;
    }
}

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    size_t threads = 256;
    size_t blocks = B;

    l1_normalization<<<blocks,threads, threads * sizeof(float)>>>(X,Y,D);
}