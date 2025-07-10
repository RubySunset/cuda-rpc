extern "C"
__global__ void f0()
{
}

extern "C"
__global__ void f1(void* ptr1)
{
}

extern "C"
__global__ void f2(void* ptr1, void* ptr2)
{
}

extern "C"
__global__ void sum_array(int* in, size_t size, int* out)
{
    int total = 0;
    for (auto i = 0; i < size; i++) {
        total += in[i];
    }
    *out = total;
}
