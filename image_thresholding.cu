#include <cuda_runtime.h>

__global__ void threshold(
    const float* __restrict__ input_image,
    float threshold_value, 
    float* __restrict__ output_image,
    size_t pixels){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < pixels){
        input_image[tid] > threshold_value ? output_image[tid] = 255 : output_image[tid] = 0;
    }

}

__global__ void threshold_f4(
    const float4* __restrict__ input_image, 
    float threshold_value, 
    float4* __restrict__ output_image, 
    size_t pixels){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < pixels){
        float4 element = input_image[tid];
        float4 temp;
        element.x > threshold_value ? temp.x = 255 : temp.x = 0;
        element.y > threshold_value ? temp.y = 255 : temp.y = 0;
        element.z > threshold_value ? temp.z = 255 : temp.z = 0;
        element.w > threshold_value ? temp.w = 255 : temp.w = 0;

        output_image[tid] = temp;
    }

}

extern "C" void solution(const float* __restrict__ input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    size_t threads = 256;
    size_t pixels = height * width;
    size_t blocks = (pixels + threads - 1) / threads;
    // threshold<<<blocks, threads>>>(input_image, threshold_value, output_image, pixels);

    const float4* f4_input_image = reinterpret_cast<const float4*>(input_image);
    float4* f4_output_image = reinterpret_cast<float4*>(output_image);
    size_t f4_pixels = pixels / 4;
    size_t f4_blocks = (f4_pixels + threads - 1) / threads;

    threshold_f4<<<f4_blocks, threads>>>(f4_input_image, threshold_value, f4_output_image, f4_pixels);

}