#include "common.h"
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <thrust/device_malloc.h>
#include <thrust/scan.h>
#include <tuple>
using namespace std;

#define NUM_THREADS 256
#define gpuErrchk(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Put any static global variables here that you will use throughout the simulation.
int blks;
int* bin_cnts;
int* bin_ptrs;
int* parts_by_bins;
int grid_side_len;
int grid_size;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bin_cnts,
                                   int grid_side_len, int* parts_by_bins) {
    // Get thread (particle) ID
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // if (tid >= grid_size)
    //     return;

    int bin_idx = blockIdx.x;
    int row = bin_idx / grid_side_len, col = bin_idx % grid_side_len;
    int parts_begin = bin_cnts[bin_idx], parts_end = bin_cnts[bin_idx + 1];
    for (int i = threadIdx.x + parts_begin; i < parts_end; i += blockDim.x) {
        auto& p = particles[parts_by_bins[i]];
        p.ax = p.ay = 0;
        for (int neighbor_row = max(row - 1, 0); neighbor_row <= min(row + 1, grid_side_len - 1);
             neighbor_row++) {
            for (int neighbor_col = max(col - 1, 0);
                 neighbor_col <= min(col + 1, grid_side_len - 1); neighbor_col++) {
                int neighbor_bin_idx = neighbor_row * grid_side_len + neighbor_col;
                for (int j = bin_cnts[neighbor_bin_idx]; j < bin_cnts[neighbor_bin_idx + 1]; j++) {
                    apply_force_gpu(p, particles[parts_by_bins[j]]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    grid_side_len = static_cast<int>(size / cutoff) + 1;
    grid_size = grid_side_len * grid_side_len;

    gpuErrchk(cudaMalloc(&bin_cnts, (grid_size + 1) * sizeof(int)));

    // gpuErrchk(cudaMemset(bin_cnts, 0, grid_size * sizeof(int)));
    // gpuErrchk(cudaMemset(bin_cnts + grid_size, num_parts, sizeof(int)));
    gpuErrchk(cudaMalloc(&bin_ptrs, grid_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&parts_by_bins, num_parts * sizeof(int)));
}

__global__ void count_bin_parts(particle_t* parts, int num_parts, int* bin_cnts,
                                int grid_side_len) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) {
        return;
    }
    auto& p = parts[tid];
    int col = p.x / cutoff, row = p.y / cutoff;
    int bin_idx = row * grid_side_len + col;
    atomicAdd(bin_cnts + bin_idx, 1);
}

void count_prefix_sum() {
    thrust::device_ptr<int> bin_cnts_ptr(bin_cnts);
    thrust::exclusive_scan(bin_cnts_ptr, bin_cnts_ptr + grid_size, bin_cnts_ptr);
    gpuErrchk(cudaMemcpy(bin_ptrs, bin_cnts, grid_size * sizeof(int), cudaMemcpyDefault));
}

__global__ void assign_parts(particle_t* parts, int num_parts, int* bin_ptrs, int* parts_by_bins,
                             int grid_side_len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) {
        return;
    }

    auto& p = parts[tid];
    int col = p.x / cutoff, row = p.y / cutoff;
    int bin_idx = row * grid_side_len + col;
    int pos = atomicAdd(bin_ptrs + bin_idx, 1);
    parts_by_bins[pos] = tid;
}

int* bin_cnts_cpu;
void init_arrays(int num_parts) {
    static bool first = true;
    if (first) {
        bin_cnts_cpu = new int[grid_size + 1]();
        bin_cnts_cpu[grid_size] = num_parts;
        first = false;
    }
    gpuErrchk(
        cudaMemcpy(bin_cnts, bin_cnts_cpu, (grid_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    init_arrays(num_parts);
    count_bin_parts<<<blks, NUM_THREADS>>>(parts, num_parts, bin_cnts, grid_side_len);
    // gpuErrchk(cudaDeviceSynchronize());
    count_prefix_sum();
    // gpuErrchk(cudaDeviceSynchronize());
    assign_parts<<<blks, NUM_THREADS>>>(parts, num_parts, bin_ptrs, parts_by_bins, grid_side_len);
    // Compute forces
    compute_forces_gpu<<<grid_size, NUM_THREADS>>>(parts, num_parts, bin_cnts, grid_side_len,
                                                   parts_by_bins);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

// Clear allocations
void clear_simulation() {
    cudaFree(bin_cnts);
    cudaFree(bin_ptrs);
    cudaFree(parts_by_bins);
    delete[] bin_cnts_cpu;
}
