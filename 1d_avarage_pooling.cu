#include <cuda_runtime.h>

__global__ void average_pooling(const float* input, float* output, size_t output_size, int stride, int padding, int kernel_size, size_t H){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < output_size){
        size_t input_index = tid * stride - padding;

        float sum = 0.0f;
        for(int i = 0; i < kernel_size; i++){
            int pos = input_index + i;
            if(pos < H){
                sum += input[pos];
            }
        }
        float average = sum / kernel_size;

        output[tid] = average;
    }
}

// Note: input, output are device pointers
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    const size_t output_size = ((H + 2*padding - kernel_size) / stride) + 1;
    size_t threads = 256;
    size_t blocks = (output_size + threads - 1) / threads;

    average_pooling<<<blocks,threads>>>(input, output, output_size, stride, padding, kernel_size, H);
}