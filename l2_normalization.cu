#include <cuda_runtime.h>

__device__ __forceinline__ size_t get_linear_index(size_t x, size_t y, size_t width){
    return x * width + y;
}

__global__ void l2_normalization(const float* __restrict__ X, float* __restrict__ Y, size_t cols){
    extern __shared__ float s_sum[];
    
    size_t row_idx = blockIdx.x;
    size_t col_idx = threadIdx.x;

    float local_sum = 0.0f;
    for(size_t i = col_idx; i < cols; i += blockDim.x){
        float pos = X[get_linear_index(row_idx, i, cols)];
        local_sum += pos * pos;
    }
    s_sum[col_idx] = local_sum;
    __syncthreads();

    for(size_t s = blockDim.x/2; s > 0 ; s>>=1){
        if(col_idx < s){
            s_sum[col_idx] += s_sum[col_idx + s];
        }
        __syncthreads();
    }

    float normalized = sqrtf(s_sum[0]);

    for(size_t i = col_idx; i < cols; i += blockDim.x){
        size_t pos = get_linear_index(row_idx, i, cols);
        Y[pos] = X[pos] / normalized;
    }
}

__global__ void l2_normalization_warp_shf(const float* __restrict__ X, float* __restrict__ Y, size_t cols){
    extern __shared__ float s_sum[];
    
    size_t row_idx = blockIdx.x;
    size_t col_idx = threadIdx.x;

    size_t lane_id = col_idx % 32;
    size_t warp_id = col_idx / 32;

    float local_sum = 0.0f;
    for(size_t i = col_idx; i < cols; i += blockDim.x){
        float pos = X[get_linear_index(row_idx, i, cols)];
        local_sum += pos * pos;
    }

    for(size_t s = 16; s > 0 ; s>>=1){
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, s);
    }

    if(lane_id == 0){
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x / 32)) ? s_sum[lane_id] : 0.0f;

        for(size_t s = 16; s > 0 ; s>>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, s);
        }

        if (lane_id == 0) s_sum[0] = val;
    }
    __syncthreads();

    float normalized = sqrtf(s_sum[0]);

    for(size_t i = col_idx; i < cols; i += blockDim.x){
        size_t pos = get_linear_index(row_idx, i, cols);
        Y[pos] = X[pos] / normalized;
    }
}

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    size_t threads = 1024;
    size_t blocks = B;

    l2_normalization_warp_shf<<<blocks,threads, threads * sizeof(float)>>>(X,Y,D);
}