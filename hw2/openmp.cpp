#include "common.h"
#include <omp.h>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <iostream>

// grid[size / cut_off][size / cut_off]
int grid_num;
std::vector<std::vector<std::unordered_set<particle_t*>>> grid;
std::vector<std::vector<omp_lock_t>> grid_locks;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    int cur_col = p.x / cutoff, cur_row = p.y / cutoff;

    if (cur_col != p.col || cur_row != p.row) {
        // first lock the cell with smaller row and col index
        if (cur_row < p.row) {
            omp_set_lock(&grid_locks[cur_row][cur_col]);
            omp_set_lock(&grid_locks[p.row][p.col]);
        } else if (cur_row > p.row) {
            omp_set_lock(&grid_locks[p.row][p.col]);
            omp_set_lock(&grid_locks[cur_row][cur_col]);
        } else {
            if (cur_col < p.col) {
                omp_set_lock(&grid_locks[cur_row][cur_col]);
                omp_set_lock(&grid_locks[p.row][p.col]);
            } else if (cur_col > p.col) {
                omp_set_lock(&grid_locks[p.row][p.col]);
                omp_set_lock(&grid_locks[cur_row][cur_col]);
            } else {
                throw "cannot be here";
            }
        }
        grid[p.row][p.col].erase(&p);
        grid[cur_row][cur_col].insert(&p);
        omp_unset_lock(&grid_locks[cur_row][cur_col]);
        omp_unset_lock(&grid_locks[p.row][p.col]);
        p.row = cur_row, p.col = cur_col;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    grid_num = int(size / cutoff) + 1;
    grid.resize(grid_num);
    for (auto &row : grid) {
        row.resize(grid_num);
    }
    grid_locks.resize(grid_num);
    for (auto &row : grid_locks) {
        row.resize(grid_num);
        for (auto &lock : row) {
            omp_init_lock(&lock);
        }
    }
    for (int i = 0; i < num_parts; ++i) {
        auto &p = parts[i];
        int col = p.x / cutoff, row = p.y / cutoff;
        p.row = row, p.col = col;
        grid[row][col].insert(&p);
    }
}

void post_simulation() {
    for (auto &row : grid_locks) {
        for (auto &lock : row) {
            omp_destroy_lock(&lock);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces
    #pragma omp parallel for collapse(2) schedule(auto)
    for (int row = 0; row < grid_num; ++row) {
        for (int col = 0; col < grid_num; ++col) {
            for (auto p : grid[row][col]) {
                p->ax = p->ay = 0;
                for (int nei_row = std::max(row - 1, 0); nei_row <= std::min(row + 1, grid_num - 1);
                     nei_row++) {
                    for (int nei_col = std::max(col - 1, 0); nei_col <= std::min(col + 1, grid_num - 1);
                         nei_col++) {
                            for (auto p_nei : grid[nei_row][nei_col]) {
                                apply_force(*p, *p_nei);
                            }
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(auto)
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
