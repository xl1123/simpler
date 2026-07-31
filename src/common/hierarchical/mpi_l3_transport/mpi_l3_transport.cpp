/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int ENVELOPE_TAG = 0x534d;

struct PendingSend {
    std::vector<uint8_t> bytes;
    MPI_Request request{MPI_REQUEST_NULL};
};

bool initialized = false;
int world_rank = -1;
int world_size = 0;
std::vector<std::unique_ptr<PendingSend>> pending_sends;

void set_error(char *buffer, uint64_t capacity, const std::string &message) {
    if (buffer == nullptr || capacity == 0) return;
    size_t count = std::min(static_cast<size_t>(capacity - 1), message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

std::string mpi_error(int code, const char *operation) {
    char detail[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(code, detail, &length);
    return std::string(operation) + " failed: " + std::string(detail, static_cast<size_t>(length));
}

void require_mpi(int code, const char *operation) {
    if (code != MPI_SUCCESS) throw std::runtime_error(mpi_error(code, operation));
}

void progress_pending() {
    for (auto it = pending_sends.begin(); it != pending_sends.end();) {
        int complete = 0;
        require_mpi(MPI_Test(&(*it)->request, &complete, MPI_STATUS_IGNORE), "MPI_Test");
        if (complete != 0) {
            it = pending_sends.erase(it);
        } else {
            ++it;
        }
    }
}

void verify_shared_topology(const std::string &topology_id) {
    if (topology_id.empty()) throw std::invalid_argument("topology_id must be non-empty");
    uint64_t local_size = topology_id.size();
    std::vector<uint64_t> sizes(static_cast<size_t>(world_size));
    require_mpi(
        MPI_Allgather(&local_size, 1, MPI_UINT64_T, sizes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
        "MPI_Allgather(topology size)"
    );
    uint64_t max_size = *std::max_element(sizes.begin(), sizes.end());
    if (max_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("topology_id exceeds MPI count range");
    }
    std::vector<char> local(static_cast<size_t>(max_size), 0);
    std::copy(topology_id.begin(), topology_id.end(), local.begin());
    std::vector<char> all(static_cast<size_t>(world_size) * static_cast<size_t>(max_size), 0);
    require_mpi(
        MPI_Allgather(
            local.data(), static_cast<int>(max_size), MPI_CHAR, all.data(), static_cast<int>(max_size), MPI_CHAR,
            MPI_COMM_WORLD
        ),
        "MPI_Allgather(topology id)"
    );
    for (int rank = 0; rank < world_size; ++rank) {
        const char *value = all.data() + static_cast<size_t>(rank) * static_cast<size_t>(max_size);
        if (std::string(value, static_cast<size_t>(sizes[rank])) != topology_id) {
            throw std::runtime_error("topology_id differs across MPI ranks");
        }
    }
}

}  // namespace

extern "C" int simpler_mpi_l3_init(
    int expected_rank, int expected_world_size, const char *topology_id, char *processor_name,
    uint64_t processor_name_capacity, char *error, uint64_t error_capacity
) {
    try {
        if (initialized) throw std::runtime_error("MPI transport is already initialized");
        int already_initialized = 0;
        require_mpi(MPI_Initialized(&already_initialized), "MPI_Initialized");
        if (already_initialized != 0) throw std::runtime_error("MPI was initialized before L3 completed local forks");
        int provided = 0;
        int argc = 0;
        char **argv = nullptr;
        require_mpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
        initialized = true;
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI implementation did not provide MPI_THREAD_FUNNELED");
        }
        require_mpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
        require_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), "MPI_Comm_rank");
        require_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
        if (world_rank != expected_rank || world_size != expected_world_size) {
            throw std::runtime_error(
                "MPI rank/world differs from launcher metadata: actual=" + std::to_string(world_rank) + "/" +
                std::to_string(world_size) + " expected=" + std::to_string(expected_rank) + "/" +
                std::to_string(expected_world_size)
            );
        }
        verify_shared_topology(topology_id == nullptr ? "" : topology_id);
        char local_name[MPI_MAX_PROCESSOR_NAME]{};
        int name_length = 0;
        require_mpi(MPI_Get_processor_name(local_name, &name_length), "MPI_Get_processor_name");
        set_error(processor_name, processor_name_capacity, std::string(local_name, static_cast<size_t>(name_length)));
        return 0;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return 1;
    }
}

extern "C" int
simpler_mpi_l3_send(int target_rank, const uint8_t *data, uint64_t size, char *error, uint64_t error_capacity) {
    try {
        if (!initialized) throw std::runtime_error("MPI transport is not initialized");
        if (target_rank < 0 || target_rank >= world_size || target_rank == world_rank) {
            throw std::invalid_argument("MPI target rank is invalid or local");
        }
        if (data == nullptr || size == 0 || size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MPI envelope size is invalid");
        }
        progress_pending();
        auto send = std::make_unique<PendingSend>();
        send->bytes.assign(data, data + size);
        require_mpi(
            MPI_Isend(
                send->bytes.data(), static_cast<int>(send->bytes.size()), MPI_BYTE, target_rank, ENVELOPE_TAG,
                MPI_COMM_WORLD, &send->request
            ),
            "MPI_Isend"
        );
        pending_sends.push_back(std::move(send));
        return 0;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return 1;
    }
}

extern "C" int simpler_mpi_l3_probe(int *source_rank, uint64_t *size, char *error, uint64_t error_capacity) {
    try {
        if (!initialized) throw std::runtime_error("MPI transport is not initialized");
        progress_pending();
        int available = 0;
        MPI_Status status{};
        require_mpi(MPI_Iprobe(MPI_ANY_SOURCE, ENVELOPE_TAG, MPI_COMM_WORLD, &available, &status), "MPI_Iprobe");
        if (available == 0) return 0;
        int count = 0;
        require_mpi(MPI_Get_count(&status, MPI_BYTE, &count), "MPI_Get_count");
        if (count <= 0) throw std::runtime_error("MPI envelope has an invalid size");
        *source_rank = status.MPI_SOURCE;
        *size = static_cast<uint64_t>(count);
        return 1;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return -1;
    }
}

extern "C" int simpler_mpi_l3_recv(
    int source_rank, uint8_t *data, uint64_t capacity, uint64_t *size, char *error, uint64_t error_capacity
) {
    try {
        if (!initialized) throw std::runtime_error("MPI transport is not initialized");
        if (data == nullptr || capacity == 0 || capacity > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MPI receive buffer is invalid");
        }
        MPI_Status status{};
        require_mpi(
            MPI_Recv(data, static_cast<int>(capacity), MPI_BYTE, source_rank, ENVELOPE_TAG, MPI_COMM_WORLD, &status),
            "MPI_Recv"
        );
        int count = 0;
        require_mpi(MPI_Get_count(&status, MPI_BYTE, &count), "MPI_Get_count");
        *size = static_cast<uint64_t>(count);
        return 0;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return 1;
    }
}

extern "C" int simpler_mpi_l3_barrier(char *error, uint64_t error_capacity) {
    try {
        if (!initialized) throw std::runtime_error("MPI transport is not initialized");
        require_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
        return 0;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return 1;
    }
}

extern "C" int simpler_mpi_l3_finalize(char *error, uint64_t error_capacity) {
    try {
        if (!initialized) return 0;
        for (auto &send : pending_sends)
            require_mpi(MPI_Wait(&send->request, MPI_STATUS_IGNORE), "MPI_Wait");
        pending_sends.clear();
        require_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(finalize)");
        require_mpi(MPI_Finalize(), "MPI_Finalize");
        initialized = false;
        world_rank = -1;
        world_size = 0;
        return 0;
    } catch (const std::exception &exc) {
        set_error(error, error_capacity, exc.what());
        return 1;
    }
}

extern "C" void simpler_mpi_l3_abort(int error_code) {
    if (initialized) MPI_Abort(MPI_COMM_WORLD, error_code);
}
