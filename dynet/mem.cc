#include "dynet/mem.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#if !_WINDOWS
#include <sys/shm.h>
#include <sys/mman.h>
#endif

#include <fcntl.h>
#if !_WINDOWS
//#include <mm_malloc.h>
#endif
#include "dynet/except.h"
#include "dynet/devices.h"
#if HAVE_CUDA
#include "dynet/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace dynet {

MemAllocator::~MemAllocator() {}

  void* aligned_malloc(size_t alignment, size_t amount) {
	assert(alignment == FIBITMAP_ALIGNMENT);
	/*
	In some rare situations, the malloc routines can return misaligned memory. 
	The routine FreeImage_Aligned_Malloc allocates a bit more memory to do
	aligned writes.  Normally, it *should* allocate "alignment" extra memory and then writes
	one dword back the true pointer.  But if the memory manager returns a
	misaligned block that is less than a dword from the next alignment, 
	then the writing back one dword will corrupt memory.

	For example, suppose that alignment is 16 and malloc returns the address 0xFFFF.

	16 - 0xFFFF % 16 + 0xFFFF = 16 - 15 + 0xFFFF = 0x10000.

	Now, you subtract one dword from that and write and that will corrupt memory.

	That's why the code below allocates *two* alignments instead of one. 
	*/
	void* mem_real = malloc(amount + 2 * alignment);
	if(!mem_real) return NULL;
	char* mem_align = (char*)((unsigned long)(2 * alignment - (unsigned long)mem_real % (unsigned long)alignment) + (unsigned long)mem_real);
	*((long*)mem_align - 1) = (long)mem_real;
	return mem_align;
}   
  
void* CPUAllocator::malloc(size_t n) {
  //void* ptr = _mm_malloc(n, align);
  void* ptr = aligned_malloc(align, n);
  if (!ptr) {
    show_pool_mem_info();
    cerr << "CPU memory allocation failed n=" << n << " align=" << align << endl;
    throw dynet::out_of_memory("CPU memory allocation failed");
  }
  return ptr;
}

void CPUAllocator::free(void* mem) {
  //_mm_free(mem);
  free(mem);
}

void CPUAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

void* SharedAllocator::malloc(size_t n) {
#if _WINDOWS
  cerr << "Shared memory not supported in Windows" << endl;
  throw dynet::out_of_memory("Shared memory allocation failed");
#else
  void* ptr = mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
  if (ptr == MAP_FAILED) {
    show_pool_mem_info();
    cerr << "Shared memory allocation failed n=" << n << endl;
    throw dynet::out_of_memory("Shared memory allocation failed");
  }
  return ptr;
#endif
}

void SharedAllocator::free(void* mem) {
//  munmap(mem, n);
}

void SharedAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

#if HAVE_CUDA
void* GPUAllocator::malloc(size_t n) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMalloc(&ptr, n));
  if (!ptr) {
    show_pool_mem_info();
    cerr << "GPU memory allocation failed n=" << n << endl;
    throw dynet::out_of_memory("GPU memory allocation failed");
  }
  return ptr;
}

void GPUAllocator::free(void* mem) {
  CUDA_CHECK(cudaFree(mem));
}

void GPUAllocator::zero(void* p, size_t n) {
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMemsetAsync(p, 0, n));
}

#endif

} // namespace dynet
