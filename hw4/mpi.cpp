#include "common.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <tuple>
#include <chrono>

// common.cpp
void particle_t::move(double size) {
    vx += ax * dt;
    vy += ay * dt;
    x += vx * dt;
    y += vy * dt;

    while (x < 0 || x > size) {
        x = x < 0 ? -x : 2 * size - x;
        vx = -vx;
    }
    while (y < 0 || y > size) {
        y = y < 0 ? -y : 2 * size - y;
        vy = -vy;
    }
}

void particle_t::apply_force(const particle_t& other) {
    double dx = other.x - x;
    double dy = other.y - y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;
    ax += coef * dx;
    ay += coef * dy;
}

void cell_t::refresh() {
    particles.splice(particles.end(), incoming_particles);
    incoming_particles.clear();
}

// Put any static global variables here that you will use throughout the simulation.
int rk_debug;
vector<vector<cell_t>> cells;
int num_rows;
int procs_x, procs_y;
int working_procs;
double cell_length;
int row_start, row_end, col_start, col_end;
int actual_row_start, actual_row_end, actual_col_start, actual_col_end;

void cell_t::move_particles(vector<vector<cell_t>>& cells, particle_t* parts, double size) {
    auto it = particles.begin();
    while (it != particles.end()) {
        auto& part = parts[(*it) - 1];
        int old_row = part.y / cell_length, old_col = part.x / cell_length;
        part.move(size);
        int new_row = part.y / cell_length, new_col = part.x / cell_length;
        if (new_row != old_row || new_col != old_col) {
            if (abs(new_row - old_row) > 1 || abs(new_col - old_col) > 1) {
                cout << rk_debug << ' ' << *it << ':' << old_row << ' ' << new_row << '\t'
                     << old_col << ' ' << new_col << endl;
            }
            cells[new_row - actual_row_start][new_col - actual_col_start]
                .incoming_particles.push_back(*it);
            it = particles.erase(it);
        } else {
            ++it;
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    rk_debug = rank;
    num_rows = int(size / cutoff) + 1;
    // cell_length = size / (double)num_rows;
    cell_length = cutoff;
    procs_x = sqrt(num_procs);
    procs_y = num_procs / procs_x;
    working_procs = procs_x * procs_y;

    if (rank >= working_procs) {
        return;
    }

    int rows_per_proc = (num_rows + procs_y - 1) / procs_y,
        cols_per_proc = (num_rows + procs_x - 1) / procs_x;

    row_start = (rank / procs_x) * rows_per_proc;
    actual_row_start = row_start;
    row_end = min(row_start + rows_per_proc, num_rows) - 1;
    actual_row_end = row_end;
    col_start = (rank % procs_x) * cols_per_proc;
    actual_col_start = col_start;
    col_end = min(col_start + cols_per_proc, num_rows) - 1;
    actual_col_end = col_end;

    if (row_start != 0) {
        actual_row_start--;
    }
    if (row_end != (num_rows - 1)) {
        actual_row_end++;
    }
    if (col_start != 0) {
        actual_col_start--;
    }
    if (col_end != (num_rows - 1)) {
        actual_col_end++;
    }

    cells.assign(actual_row_end - actual_row_start + 1,
                 vector<cell_t>(actual_col_end - actual_col_start + 1));
    for (int i = 0; i < num_parts; i++) {
        auto& part = parts[i];
        int row = part.y / cell_length, col = part.x / cell_length;
        if (row >= row_start && row <= row_end && col >= col_start && col <= col_end) {
            cells[row - actual_row_start][col - actual_col_start].particles.push_back(part.id);
        }
    }
}

vector<particle_t> prepare_send_particles(particle_t* parts, vector<int> rows, vector<int> cols,
                                          bool is_incoming) {
    vector<particle_t> send_particles;
    for (int i = 0; i < rows.size(); i++) {
        auto& cell = cells[rows[i]][cols[i]];
        auto& particles = is_incoming ? cell.incoming_particles : cell.particles;
        for (int part_idx : particles) {
            send_particles.push_back(parts[part_idx - 1]);
        }
        if (is_incoming) {
            particles.clear();
        }
    }
    return send_particles;
}

#define MPI_CHECK(call)                                                                            \
    do {                                                                                           \
        int error_code = call;                                                                     \
        if (error_code != MPI_SUCCESS) {                                                           \
            char error_string[MPI_MAX_ERROR_STRING];                                               \
            int length;                                                                            \
            MPI_Error_string(error_code, error_string, &length);                                   \
            fprintf(stderr, "MPI error at %s:%d - %s\n", __FILE__, __LINE__, error_string);        \
            MPI_Abort(MPI_COMM_WORLD, error_code);                                                 \
        }                                                                                          \
    } while (0)

// vector<double> simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    // exchange particles with neighboring processors

    if (rank >= working_procs) {
        return;
    }

    vector<particle_t> send_buf, recv_buf;
    vector<int> rows, cols;

    auto log_send = [](int src, int dst, int cnt) {
        return;
        if (cnt != 0)
            cout << src << " -> " << dst << ": " << cnt << endl;
    };

    auto log_recv = [](int src, int dst, int cnt) {
        return;
        if (cnt != 0)
            cout << dst << " <- " << src << ": " << cnt << endl;
    };

    // vector<double> ret_time;
    // auto start_time = std::chrono::steady_clock::now();
    // upwards
    if (row_end != num_rows - 1) {
        rows.assign(col_end - col_start + 1, row_end - actual_row_start);
        cols.resize(col_end - col_start + 1);
        iota(cols.begin(), cols.end(), col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_end != num_rows - 1) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        for (auto& cell : cells[0]) {
            cell.particles.clear();
        }
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[0][part.x / cell_length - actual_col_start].particles.push_back(part.id);
        }
    }

    // downwards
    if (row_start != 0) {
        if (rows.size() != col_end - col_start + 1) {
            rows.assign(col_end - col_start + 1, row_start - actual_row_start);
        } else {
            fill(rows.begin(), rows.end(), row_start - actual_row_start);
        }
        if (cols.size() != col_end - col_start + 1) {
            cols.resize(col_end - col_start + 1);
        }
        iota(cols.begin(), cols.end(), col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        for (auto& cell : cells[actual_row_end - actual_row_start]) {
            cell.particles.clear();
        }
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[actual_row_end - actual_row_start][part.x / cell_length - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // left
    if (col_start != 0) {
        rows.resize(row_end - row_start + 1);
        iota(rows.begin(), rows.end(), row_start - actual_row_start);
        cols.assign(row_end - row_start + 1, col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD));
    }
    if (col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (col_start != 0) {
        MPI_CHECK(
            MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD));
    }
    if (col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
        for (int row = row_start; row <= row_end; row++) {
            cells[row - actual_row_start][actual_col_end - actual_col_start].particles.clear();
        }
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[part.y / cell_length - actual_row_start][actual_col_end - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // right
    if (col_end != (num_rows - 1)) {
        if (cols.size() != row_end - row_start + 1) {
            cols.assign(row_end - row_start + 1, col_end - actual_col_start);
        } else {
            fill(cols.begin(), cols.end(), col_end - actual_col_start);
        }
        if (rows.size() != row_end - row_start + 1) {
            rows.resize(row_end - row_start + 1);
        }
        iota(rows.begin(), rows.end(), row_start - actual_row_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD));
    }
    if (col_start != 0) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (col_end != (num_rows - 1)) {
        MPI_CHECK(
            MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD));
    }
    if (col_start != 0) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
        for (int row = row_start; row <= row_end; row++) {
            cells[row - actual_row_start][0].particles.clear();
        }
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[part.y / cell_length - actual_row_start][0].particles.push_back(part.id);
        }
    }

    // upleft
    if (row_end != (num_rows - 1) && col_start != 0) {
        rows.assign(1, row_end - actual_row_start);
        cols.assign(1, col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x - 1, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x - 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x + 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cells[0][actual_col_end - actual_col_start].particles.clear();
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[0][actual_col_end - actual_col_start].particles.push_back(part.id);
        }
    }

    // downright
    if (row_start != 0 && col_end != (num_rows - 1)) {
        rows.assign(1, row_start - actual_row_start);
        cols.assign(1, col_end - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x + 1, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x + 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x - 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cells[actual_row_end - actual_row_start][0].particles.clear();
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[actual_row_end - actual_row_start][0].particles.push_back(part.id);
        }
    }

    // upright
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        rows.assign(1, row_end - actual_row_start);
        cols.assign(1, col_end - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x + 1, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_start != 0) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x + 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_start != 0) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x - 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cells[0][0].particles.clear();
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[0][0].particles.push_back(part.id);
        }
    }

    // downleft
    if (row_start != 0 && col_start != 0) {
        rows.assign(1, row_start - actual_row_start);
        cols.assign(1, col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, false));
        int cnt = send_buf.size();
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x - 1, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        recv_buf.resize(cnt);
    }
    if (row_start != 0 && col_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x - 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x + 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        cells[actual_row_end - actual_row_start][actual_col_end - actual_col_start]
            .particles.clear();
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[actual_row_end - actual_row_start][actual_col_end - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // auto end_time = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end_time - start_time;
    // ret_time.push_back(diff.count());

    // start_time = end_time;

    for (int row = row_start; row <= row_end; row++) {
        for (int col = col_start; col <= col_end; col++) {
            for (int p_idx : cells[row - actual_row_start][col - actual_col_start].particles) {
                auto& p = parts[p_idx - 1];
                p.ax = p.ay = 0;
                for (int nei_row = max(row - 1, 0); nei_row <= min(row + 1, num_rows - 1);
                     nei_row++) {
                    for (int nei_col = max(col - 1, 0); nei_col <= min(col + 1, num_rows - 1);
                         nei_col++) {
                        for (int q : cells[nei_row - actual_row_start][nei_col - actual_col_start]
                                         .particles) {
                            p.apply_force(parts[q - 1]);
                        }
                    }
                }
            }
        }
    }

    for (int row = row_start; row <= row_end; row++) {
        for (int col = col_start; col <= col_end; col++) {
            cells[row - actual_row_start][col - actual_col_start].move_particles(cells, parts,
                                                                                 size);
        }
    }

    for (int row = row_start; row <= row_end; row++) {
        for (int col = col_start; col <= col_end; col++) {
            cells[row - actual_row_start][col - actual_col_start].refresh();
        }
    }
    // end_time = std::chrono::steady_clock::now();
    // diff = end_time - start_time;
    // ret_time.push_back(diff.count());
    // start_time = end_time;

    // upwards
    if (row_end != num_rows - 1) {
        rows.assign(col_end - col_start + 1, actual_row_end - actual_row_start);
        cols.resize(col_end - col_start + 1);
        iota(cols.begin(), cols.end(), col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank + procs_x, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank - procs_x, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_end != num_rows - 1) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_start - actual_row_start][part.x / cell_length - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // downwards
    if (row_start != 0) {
        if (rows.size() != col_end - col_start + 1) {
            rows.assign(col_end - col_start + 1, 0);
        } else {
            fill(rows.begin(), rows.end(), 0);
        }
        if (cols.size() != col_end - col_start + 1) {
            cols.resize(col_end - col_start + 1);
        }
        iota(cols.begin(), cols.end(), col_start - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank - procs_x, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank + procs_x, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1)) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_end - actual_row_start][part.x / cell_length - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // left
    if (col_start != 0) {
        rows.resize(row_end - row_start + 1);
        iota(rows.begin(), rows.end(), row_start - actual_row_start);
        cols.assign(row_end - row_start + 1, 0);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank - 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD));
    }
    if (col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank + 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (col_start != 0) {
        MPI_CHECK(
            MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD));
    }
    if (col_end != (num_rows - 1)) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
                           &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[part.y / cell_length - actual_row_start][col_end - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // right
    if (col_end != (num_rows - 1)) {
        if (cols.size() != row_end - row_start + 1) {
            cols.assign(row_end - row_start + 1, actual_col_end - actual_col_start);
        } else {
            fill(cols.begin(), cols.end(), actual_col_end - actual_col_start);
        }
        if (rows.size() != row_end - row_start + 1) {
            rows.resize(row_end - row_start + 1);
        }
        iota(rows.begin(), rows.end(), row_start - actual_row_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank + 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD));
    }
    if (col_start != 0) {
        int cnt;
        MPI_CHECK(MPI_Recv(&cnt, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank - 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (col_end != (num_rows - 1)) {
        MPI_CHECK(
            MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD));
    }
    if (col_start != 0) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD,
                           &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[part.y / cell_length - actual_row_start][col_start - actual_col_start]
                .particles.push_back(part.id);
        }
    }

    // upleft
    if (row_end != (num_rows - 1) && col_start != 0) {
        rows.assign(1, actual_row_end - actual_row_start);
        cols.assign(1, 0);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank + procs_x - 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x - 1, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank - procs_x + 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x - 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x + 1, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_start - actual_row_start][col_end - actual_col_start].particles.push_back(
                part.id);
        }
    }

    // downright
    if (row_start != 0 && col_end != (num_rows - 1)) {
        rows.assign(1, 0);
        cols.assign(1, actual_col_end - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank - procs_x + 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x + 1, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank + procs_x - 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_start != 0 && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x + 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_start != 0) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x - 1, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_end - actual_row_start][col_start - actual_col_start].particles.push_back(
                part.id);
        }
    }

    // upright
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        rows.assign(1, actual_row_end - actual_row_start);
        cols.assign(1, actual_col_end - actual_col_start);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank + procs_x + 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank + procs_x + 1, 0, MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_start != 0) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank - procs_x - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank - procs_x - 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank + procs_x + 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_start != 0 && col_start != 0) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank - procs_x - 1, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_start - actual_row_start][col_start - actual_col_start].particles.push_back(
                part.id);
        }
    }

    // downleft
    if (row_start != 0 && col_start != 0) {
        rows.assign(1, 0);
        cols.assign(1, 0);
        send_buf = move(prepare_send_particles(parts, rows, cols, true));
        int cnt = send_buf.size();
        log_send(rank, rank - procs_x - 1, cnt);
        MPI_CHECK(MPI_Send(&cnt, 1, MPI_INT, rank - procs_x - 1, 0, MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        int cnt;
        MPI_CHECK(
            MPI_Recv(&cnt, 1, MPI_INT, rank + procs_x + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        log_recv(rank + procs_x + 1, rank, cnt);
        recv_buf.resize(cnt);
    }
    if (row_start != 0 && col_start != 0) {
        MPI_CHECK(MPI_Send(send_buf.data(), send_buf.size(), PARTICLE, rank - procs_x - 1, 0,
                           MPI_COMM_WORLD));
    }
    if (row_end != (num_rows - 1) && col_end != (num_rows - 1)) {
        MPI_Status mpi_status;
        MPI_CHECK(MPI_Recv(recv_buf.data(), recv_buf.size(), PARTICLE, rank + procs_x + 1, 0,
                           MPI_COMM_WORLD, &mpi_status));
        int cnt;
        MPI_Get_count(&mpi_status, PARTICLE, &cnt);
        assert(cnt == recv_buf.size());
        for (auto& part : recv_buf) {
            parts[part.id - 1] = part;
            cells[row_end - actual_row_start][col_end - actual_col_start].particles.push_back(
                part.id);
        }
    }
    // end_time = std::chrono::steady_clock::now();
    // diff = end_time - start_time;
    // ret_time.push_back(diff.count());
    // return ret_time;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // first gather particle numbers from all processes

    vector<particle_t> local_particles;
    if (rank < working_procs) {
        for (int row = row_start; row <= row_end; row++) {
            for (int col = col_start; col <= col_end; col++) {
                for (int p_idx : cells[row - actual_row_start][col - actual_col_start].particles) {
                    local_particles.push_back(parts[p_idx - 1]);
                }
            }
        }
    }
    int local_particles_cnt = local_particles.size();
    vector<int> particle_cnts;
    if (rank == 0) {
        particle_cnts.resize(num_procs);
    }
    MPI_Gather(&local_particles_cnt, 1, MPI_INT, particle_cnts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    vector<int> displs;
    if (rank == 0) {
        displs.resize(num_procs);
        partial_sum(particle_cnts.begin(), particle_cnts.end() - 1, displs.begin() + 1);
    }
    vector<particle_t> parts_for_save;
    if (rank == 0) {
        parts_for_save.resize(num_parts);
    }
    // then gather particles from all processes
    MPI_Gatherv(local_particles.data(), local_particles_cnt, PARTICLE, parts_for_save.data(),
                particle_cnts.data(), displs.data(), PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (auto& part : parts_for_save) {
            parts[part.id - 1] = part;
        }
    }
}