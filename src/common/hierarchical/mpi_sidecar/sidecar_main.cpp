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

#include "sidecar_protocol.h"

#include <mpi.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sidecar = simpler::mpi_sidecar;

namespace {

constexpr int ENVELOPE_TAG = 0x534c;

struct Options {
    std::string proxy_socket_template;
    std::string topology_id;
    std::string worker_map;
    bool manage_proxy{false};
    std::string proxy_python;
    std::string proxy_session_dir_template;
    std::string bootstrap_socket;
    int proxy_startup_timeout_ms{30000};
};

struct PendingSend {
    std::vector<uint8_t> bytes;
    MPI_Request request{MPI_REQUEST_NULL};
};

std::string require_value(int argc, char **argv, int &index, const char *name) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
    return argv[++index];
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--proxy-socket-template") {
            options.proxy_socket_template = require_value(argc, argv, i, "--proxy-socket-template");
        } else if (arg == "--topology-id") {
            options.topology_id = require_value(argc, argv, i, "--topology-id");
        } else if (arg == "--worker-map") {
            options.worker_map = require_value(argc, argv, i, "--worker-map");
        } else if (arg == "--manage-proxy") {
            options.manage_proxy = true;
        } else if (arg == "--proxy-python") {
            options.proxy_python = require_value(argc, argv, i, "--proxy-python");
        } else if (arg == "--proxy-session-dir-template") {
            options.proxy_session_dir_template = require_value(argc, argv, i, "--proxy-session-dir-template");
        } else if (arg == "--bootstrap-socket") {
            options.bootstrap_socket = require_value(argc, argv, i, "--bootstrap-socket");
        } else if (arg == "--proxy-startup-timeout-ms") {
            options.proxy_startup_timeout_ms = std::stoi(require_value(argc, argv, i, "--proxy-startup-timeout-ms"));
        } else if (arg == "--help") {
            std::cout << "Usage: simpler-mpi-l4-sidecar --proxy-socket-template PATH_WITH_%r "
                         "--topology-id ID --worker-map MAP [--manage-proxy --proxy-python PYTHON "
                         "--proxy-session-dir-template PATH_WITH_%r --bootstrap-socket PATH]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.proxy_socket_template.empty()) throw std::invalid_argument("--proxy-socket-template is required");
    if (options.proxy_socket_template.find("%r") == std::string::npos) {
        throw std::invalid_argument("--proxy-socket-template must contain %r");
    }
    if (options.topology_id.empty()) throw std::invalid_argument("--topology-id is required");
    if (options.worker_map.empty()) throw std::invalid_argument("--worker-map is required");
    if (options.proxy_startup_timeout_ms <= 0) {
        throw std::invalid_argument("--proxy-startup-timeout-ms must be positive");
    }
    if (options.manage_proxy) {
        if (options.proxy_python.empty()) throw std::invalid_argument("--proxy-python is required with --manage-proxy");
        if (options.proxy_session_dir_template.empty() ||
            options.proxy_session_dir_template.find("%r") == std::string::npos) {
            throw std::invalid_argument("--proxy-session-dir-template with %r is required with --manage-proxy");
        }
        if (options.bootstrap_socket.empty()) {
            throw std::invalid_argument("--bootstrap-socket is required with --manage-proxy");
        }
    }
    return options;
}

std::string rank_path(const std::string &path_template, int rank) {
    std::string result = path_template;
    size_t pos = result.find("%r");
    result.replace(pos, 2, std::to_string(rank));
    return result;
}

int try_connect_proxy(const std::string &path, int *error) {
    sockaddr_un address{};
    if (path.size() >= sizeof(address.sun_path)) throw std::invalid_argument("proxy socket path is too long");
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("proxy socket failed: ") + std::strerror(errno));
    if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        *error = errno;
        ::close(fd);
        return -1;
    }
    return fd;
}

int connect_proxy(const std::string &path) {
    int error = 0;
    int fd = try_connect_proxy(path, &error);
    if (fd < 0) throw std::runtime_error("proxy connect failed for " + path + ": " + std::strerror(error));
    return fd;
}

std::string json_escape(const std::string &value) {
    std::string result;
    for (char c : value) {
        if (c == '"' || c == '\\') result.push_back('\\');
        if (c >= 0x20) result.push_back(c);
    }
    return result;
}

std::string gather_world_json(int rank, int world_size, const std::string &worker_map) {
    char local_name[MPI_MAX_PROCESSOR_NAME]{};
    int local_length = 0;
    if (MPI_Get_processor_name(local_name, &local_length) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Get_processor_name failed");
    }
    std::vector<char> names(static_cast<size_t>(world_size) * MPI_MAX_PROCESSOR_NAME, 0);
    if (MPI_Allgather(
            local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD
        ) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allgather(hostname) failed");
    }
    std::string json = "{\"event\":\"MPI world READY\",\"rank\":" + std::to_string(rank) +
                       ",\"world_size\":" + std::to_string(world_size) + ",\"worker_map\":\"" +
                       json_escape(worker_map) + "\",\"hosts\":[";
    for (int i = 0; i < world_size; ++i) {
        if (i != 0) json += ',';
        const char *name = names.data() + static_cast<size_t>(i) * MPI_MAX_PROCESSOR_NAME;
        json += "\"" + json_escape(name) + "\"";
    }
    json += "]}";
    return json;
}

void verify_shared_string(const std::string &value, const char *label, int world_size) {
    uint64_t local_size = value.size();
    std::vector<uint64_t> sizes(static_cast<size_t>(world_size));
    MPI_Allgather(&local_size, 1, MPI_UINT64_T, sizes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    uint64_t max_size = *std::max_element(sizes.begin(), sizes.end());
    std::vector<char> local(static_cast<size_t>(max_size), 0);
    std::copy(value.begin(), value.end(), local.begin());
    std::vector<char> all(static_cast<size_t>(world_size) * static_cast<size_t>(max_size), 0);
    MPI_Allgather(
        local.data(), static_cast<int>(max_size), MPI_CHAR, all.data(), static_cast<int>(max_size), MPI_CHAR,
        MPI_COMM_WORLD
    );
    for (int rank = 0; rank < world_size; ++rank) {
        std::string peer(all.data() + static_cast<size_t>(rank) * max_size, static_cast<size_t>(sizes[rank]));
        if (peer != value) throw std::runtime_error(std::string(label) + " differs across MPI ranks");
    }
}

int parse_nonnegative(const std::string &value, const char *label) {
    if (value.empty()) throw std::invalid_argument(std::string(label) + " is empty");
    size_t consumed = 0;
    long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(label) + " is not a non-negative int");
    }
    return static_cast<int>(parsed);
}

void validate_worker_map(const std::string &worker_map, int world_size) {
    std::set<int> ranks;
    std::set<int> workers;
    size_t entry_start = 0;
    while (entry_start <= worker_map.size()) {
        size_t entry_end = worker_map.find(';', entry_start);
        std::string entry = worker_map.substr(entry_start, entry_end - entry_start);
        size_t separator = entry.find(':');
        if (separator == std::string::npos || entry.find(':', separator + 1) != std::string::npos) {
            throw std::invalid_argument("worker map entry must use rank:worker,worker");
        }
        int rank = parse_nonnegative(entry.substr(0, separator), "worker map rank");
        if (rank >= world_size || !ranks.insert(rank).second) {
            throw std::invalid_argument("worker map rank is out of range or duplicated");
        }
        std::string worker_list = entry.substr(separator + 1);
        size_t worker_start = 0;
        while (!worker_list.empty() && worker_start <= worker_list.size()) {
            size_t worker_end = worker_list.find(',', worker_start);
            int worker = parse_nonnegative(worker_list.substr(worker_start, worker_end - worker_start), "worker id");
            if (!workers.insert(worker).second) throw std::invalid_argument("worker id is assigned more than once");
            if (worker_end == std::string::npos) break;
            worker_start = worker_end + 1;
        }
        if (entry_end == std::string::npos) break;
        entry_start = entry_end + 1;
    }
    if (ranks.size() != static_cast<size_t>(world_size)) {
        throw std::invalid_argument("worker map must contain every MPI rank exactly once");
    }
}

std::string worker_ids_for_rank(const std::string &worker_map, int target_rank) {
    size_t entry_start = 0;
    while (entry_start <= worker_map.size()) {
        size_t entry_end = worker_map.find(';', entry_start);
        std::string entry = worker_map.substr(entry_start, entry_end - entry_start);
        size_t separator = entry.find(':');
        if (separator == std::string::npos) throw std::invalid_argument("worker map entry is missing ':'");
        if (parse_nonnegative(entry.substr(0, separator), "worker map rank") == target_rank) {
            return entry.substr(separator + 1);
        }
        if (entry_end == std::string::npos) break;
        entry_start = entry_end + 1;
    }
    throw std::invalid_argument("launcher rank is missing from worker map");
}

pid_t start_proxy(const Options &options, int rank) {
    std::vector<std::string> arguments = {
        options.proxy_python,
        "-m",
        "simpler.remote_l3_sidecar_proxy",
        "--rank",
        std::to_string(rank),
        "--sidecar-socket",
        rank_path(options.proxy_socket_template, rank),
        "--session-dir",
        rank_path(options.proxy_session_dir_template, rank),
        "--worker-ids",
        worker_ids_for_rank(options.worker_map, rank),
    };
    if (rank == 0) {
        arguments.push_back("--bootstrap-socket");
        arguments.push_back(options.bootstrap_socket);
    }
    pid_t pid = ::fork();
    if (pid < 0) throw std::runtime_error(std::string("proxy fork failed: ") + std::strerror(errno));
    if (pid == 0) {
        if (::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || ::getppid() == 1) {
            std::_Exit(126);
        }
        std::vector<char *> argv;
        argv.reserve(arguments.size() + 1);
        for (std::string &argument : arguments)
            argv.push_back(argument.data());
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        std::cerr << "MPI rank-local proxy exec failed: " << std::strerror(errno) << std::endl;
        std::_Exit(127);
    }
    return pid;
}

void terminate_proxy(pid_t &pid) {
    if (pid <= 0) return;
    if (::kill(pid, SIGTERM) != 0 && errno != ESRCH) {
        std::cerr << "proxy SIGTERM failed: " << std::strerror(errno) << std::endl;
    }
    for (int attempt = 0; attempt < 100; ++attempt) {
        int status = 0;
        pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) {
            pid = -1;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (::kill(pid, SIGKILL) != 0 && errno != ESRCH) {
        std::cerr << "proxy SIGKILL failed: " << std::strerror(errno) << std::endl;
    }
    while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    pid = -1;
}

int connect_managed_proxy(const std::string &path, pid_t &pid, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    int error = ENOENT;
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = try_connect_proxy(path, &error);
        if (fd >= 0) return fd;
        int status = 0;
        pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            pid = -1;
            throw std::runtime_error("rank-local proxy exited before accepting the sidecar connection");
        }
        if (result < 0 && errno != EINTR) {
            throw std::runtime_error(std::string("proxy waitpid failed: ") + std::strerror(errno));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("rank-local proxy did not listen at " + path + ": " + std::strerror(error));
}

void wait_managed_proxy(pid_t &pid, const std::string &proxy_path, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            pid = -1;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                throw std::runtime_error("rank-local proxy exited unsuccessfully");
            }
            if (::access(proxy_path.c_str(), F_OK) == 0) {
                throw std::runtime_error("rank-local proxy left its Unix socket behind");
            }
            return;
        }
        if (result < 0 && errno != EINTR) {
            throw std::runtime_error(std::string("proxy waitpid failed: ") + std::strerror(errno));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    terminate_proxy(pid);
    throw std::runtime_error("rank-local proxy did not exit within the cleanup deadline");
}

void progress_sends(std::vector<std::unique_ptr<PendingSend>> &pending) {
    for (auto it = pending.begin(); it != pending.end();) {
        int complete = 0;
        MPI_Test(&(*it)->request, &complete, MPI_STATUS_IGNORE);
        if (complete) {
            it = pending.erase(it);
        } else {
            ++it;
        }
    }
}

void start_send(std::vector<std::unique_ptr<PendingSend>> &pending, const sidecar::Envelope &envelope) {
    auto send = std::make_unique<PendingSend>();
    send->bytes = sidecar::encode(envelope);
    if (send->bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI envelope exceeds MPI int count");
    }
    MPI_Isend(
        send->bytes.data(), static_cast<int>(send->bytes.size()), MPI_BYTE, envelope.target_rank, ENVELOPE_TAG,
        MPI_COMM_WORLD, &send->request
    );
    pending.push_back(std::move(send));
}

void wait_sends(std::vector<std::unique_ptr<PendingSend>> &pending) {
    for (auto &send : pending)
        MPI_Wait(&send->request, MPI_STATUS_IGNORE);
    pending.clear();
}

int run_loop(int proxy_fd, int rank, int world_size, const std::string &world_json) {
    sidecar::Envelope ready;
    ready.type = sidecar::MessageType::WORLD_READY;
    ready.source_rank = rank;
    ready.target_rank = rank;
    ready.payload.assign(world_json.begin(), world_json.end());
    sidecar::send_local(proxy_fd, ready);
    std::cout << world_json << std::endl;

    std::vector<std::unique_ptr<PendingSend>> pending;
    bool running = true;
    while (running) {
        progress_sends(pending);
        int available = 0;
        MPI_Status status{};
        do {
            MPI_Iprobe(MPI_ANY_SOURCE, ENVELOPE_TAG, MPI_COMM_WORLD, &available, &status);
            if (!available) break;
            int count = 0;
            MPI_Get_count(&status, MPI_BYTE, &count);
            if (count < 0 || static_cast<size_t>(count) > sidecar::HEADER_BYTES + sidecar::MAX_PAYLOAD_BYTES) {
                throw std::runtime_error("MPI envelope has invalid size");
            }
            std::vector<uint8_t> bytes(static_cast<size_t>(count));
            MPI_Recv(bytes.data(), count, MPI_BYTE, status.MPI_SOURCE, ENVELOPE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sidecar::Envelope envelope = sidecar::decode(bytes.data(), bytes.size());
            if (envelope.source_rank != status.MPI_SOURCE || envelope.target_rank != rank) {
                throw std::runtime_error("MPI envelope rank metadata mismatch");
            }
            sidecar::send_local(proxy_fd, envelope);
            if (envelope.type == sidecar::MessageType::SHUTDOWN) running = false;
        } while (running);
        if (!running) break;

        pollfd descriptor{proxy_fd, POLLIN, 0};
        int poll_result = ::poll(&descriptor, 1, 10);
        if (poll_result < 0 && errno == EINTR) continue;
        if (poll_result < 0) throw std::runtime_error(std::string("proxy poll failed: ") + std::strerror(errno));
        if (poll_result == 0) continue;
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            throw std::runtime_error("local proxy disconnected");
        }
        if ((descriptor.revents & POLLIN) == 0) continue;
        sidecar::Envelope envelope = sidecar::recv_local(proxy_fd);
        if (envelope.source_rank != -1 && envelope.source_rank != rank) {
            throw std::runtime_error("local proxy supplied a foreign source rank");
        }
        envelope.source_rank = rank;
        if (envelope.type == sidecar::MessageType::SHUTDOWN) {
            if (rank != 0) throw std::runtime_error("only rank 0 may stop the MPI world");
            for (int target = 1; target < world_size; ++target) {
                envelope.target_rank = target;
                start_send(pending, envelope);
            }
            envelope.target_rank = rank;
            sidecar::send_local(proxy_fd, envelope);
            running = false;
        } else if (envelope.target_rank == rank) {
            sidecar::send_local(proxy_fd, envelope);
        } else {
            if (envelope.target_rank < 0 || envelope.target_rank >= world_size) {
                throw std::runtime_error("local proxy selected an invalid target rank");
            }
            start_send(pending, envelope);
        }
    }
    wait_sends(pending);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    int proxy_fd = -1;
    pid_t proxy_pid = -1;
    bool mpi_initialized = false;
    try {
        Options options = parse_options(argc, argv);
        const char *rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
        if (rank_env == nullptr) rank_env = std::getenv("PMI_RANK");
        if (rank_env == nullptr) {
            throw std::runtime_error("MPI launcher rank environment is unavailable before MPI_Init");
        }
        int launcher_rank = std::stoi(rank_env);
        std::string proxy_path = rank_path(options.proxy_socket_template, launcher_rank);
        if (options.manage_proxy) {
            proxy_pid = start_proxy(options, launcher_rank);
            proxy_fd = connect_managed_proxy(proxy_path, proxy_pid, options.proxy_startup_timeout_ms);
        } else {
            proxy_fd = connect_proxy(proxy_path);
        }

        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed");
        }
        mpi_initialized = true;
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI implementation lacks MPI_THREAD_FUNNELED");
        int rank = -1;
        int world_size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (rank != launcher_rank) throw std::runtime_error("launcher rank and MPI rank differ");
        verify_shared_string(options.topology_id, "topology id", world_size);
        verify_shared_string(options.worker_map, "worker map", world_size);
        validate_worker_map(options.worker_map, world_size);
        std::string world_json = gather_world_json(rank, world_size, options.worker_map);
        int result = run_loop(proxy_fd, rank, world_size, world_json);
        ::close(proxy_fd);
        proxy_fd = -1;
        MPI_Finalize();
        mpi_initialized = false;
        if (options.manage_proxy) {
            wait_managed_proxy(proxy_pid, proxy_path, options.proxy_startup_timeout_ms);
            std::cout << "MPI rank " << rank << " proxy cleanup PASS" << std::endl;
        }
        return result;
    } catch (const std::exception &error) {
        std::cerr << "simpler MPI sidecar failed: " << error.what() << std::endl;
        if (proxy_fd >= 0) ::close(proxy_fd);
        terminate_proxy(proxy_pid);
        if (mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
