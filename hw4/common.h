#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdint>
#include <iostream>
#include <list>
#include <mpi.h>
#include <vector>
using namespace std;

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
class particle_t {
  public:
    uint64_t id; // Particle ID
    double x;    // Position X
    double y;    // Position Y
    double vx;   // Velocity X
    double vy;   // Velocity Y
    double ax;   // Acceleration X
    double ay;   // Acceleration Y

    bool operator==(const particle_t& other) const { return id == other.id && x == other.x && y == other.y; }
    void move(double size);
    void apply_force(const particle_t& other);
    friend ostream& operator<<(ostream& os, particle_t& part) {
        os << "id: " << part.id << "; row: " << part.y / cutoff << "; col: " << part.x / cutoff
           << endl;
        return os;
    }
};

class cell_t {
  public:
    list<int> particles;
    list<int> incoming_particles;

    void refresh();
    void move_particles(vector<vector<cell_t>>& cells, particle_t* parts, double size);
};

extern MPI_Datatype PARTICLE;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs);

#endif