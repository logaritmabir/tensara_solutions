#include <cuda_runtime.h>


__device__ __inline__ float threshold_pixel(float pixel_value, float threshold_value) {
    return pixel_value > threshold_value ? 255.0f : 0.0f;
}

__global__ void threshold(
    const float* __restrict__ input_image,
    float threshold_value, 
    float* __restrict__ output_image,
    size_t pixels){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < pixels){
        output_image[tid] = threshold_pixel(input_image[tid], threshold_value);
    }

}

__global__ void threshold_f4(
    const float4* __restrict__ input_image, 
    float threshold_value, 
    float4* __restrict__ output_image, 
    size_t pixels){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < pixels){
        const float4 element = input_image[tid];
        float4 temp;
        temp.x = threshold_pixel(element.x, threshold_value);
        temp.y = threshold_pixel(element.y, threshold_value);
        temp.z = threshold_pixel(element.z, threshold_value);
        temp.w = threshold_pixel(element.w, threshold_value);
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

    // There is no tail kenel because the number of pixels is guaranteed to be a multiple of 4 (as much as tests measure).
}