#include <stdio.h>
#include <utility>

using namespace std;

template <unsigned int remainder>
__global__ void dotProductKernel(unsigned int n, float *dot, float *vec1, float *vec2)
{
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    extern __shared__ volatile float cache[];

    float temp = 0.0;

    if(remainder & 1 == 1){
    temp += vec1[index] * vec2[index];
    index += stride;
    }if(remainder & 2 == 1){
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    }if(remainder & 4 == 1){
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    }
    while(index < n){
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    temp += vec1[index] * vec2[index];
    index += stride;
    }

    cache[tid] = temp;
    __syncthreads();

    // reduction
    for (unsigned int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
    }

    if (tid == 0)
    {
        atomicAdd(dot, cache[0]);
    }
}

__host__ pair<float, float> dotProduct(const int n, const float *const h_vec1, const float *const h_vec2)
{
    unsigned int NUM_BLOCKS = n / 512 / 16, NUM_THREADS = 512;
    float *d_vec1, *d_vec2, *d_result;
    float *h_result = new float;
    float result, exec_time;

    cudaMalloc((void **)&d_vec1, n * sizeof(float));
    cudaMalloc((void **)&d_vec2, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_vec1, h_vec1, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0.0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    switch ((n / NUM_BLOCKS / NUM_THREADS) % 8)
    {
    case 0:
        dotProductKernel<0><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 1:
        dotProductKernel<1><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 2:
        dotProductKernel<2><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 3:
        dotProductKernel<3><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 4:
        dotProductKernel<4><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 5:
        dotProductKernel<5><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 6:
        dotProductKernel<6><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    case 7:
        dotProductKernel<7><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(n, d_result, d_vec1, d_vec2);
        break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    result = *h_result;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete h_result;
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);

    return {result, exec_time};
}

int main()
{
    const unsigned int n = 1 << 30;
    float *h_vec1 = new float[n];
    float *h_vec2 = new float[n];
    pair<float, float> res;

    for (unsigned int i = 0; i < n; i++)
    {
        h_vec1[i] = 0.0003 * (i & 412);
        h_vec2[i] = 0.00008 * (i & 3199);
    }

    res = dotProduct(n, h_vec1, h_vec2);

    printf("%d elements dotproduct\n\nresult: %f\nexec time: %fms\n", n, res.first, res.second);
    delete h_vec1;
    delete h_vec2;

    return 0;
}