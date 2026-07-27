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

#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace simpler::mpi_sidecar {
namespace {

constexpr uint8_t MAGIC[4] = {'S', 'L', 'M', '1'};

void put_u32(std::vector<uint8_t> &out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xffU));
}

void put_u64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xffU));
}

uint32_t get_u32(const uint8_t *data) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(data[i]) << (8 * i);
    return value;
}

uint64_t get_u64(const uint8_t *data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(data[i]) << (8 * i);
    return value;
}

void write_all(int fd, const uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
#ifdef MSG_NOSIGNAL
        ssize_t written = ::send(fd, data + offset, size - offset, MSG_NOSIGNAL);
#else
        ssize_t written = ::send(fd, data + offset, size - offset, 0);
#endif
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            throw std::runtime_error(std::string("sidecar local send failed: ") + std::strerror(errno));
        }
        offset += static_cast<size_t>(written);
    }
}

void read_all(int fd, uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        ssize_t received = ::recv(fd, data + offset, size - offset, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received < 0) {
            throw std::runtime_error(std::string("sidecar local recv failed: ") + std::strerror(errno));
        }
        if (received == 0) throw std::runtime_error("sidecar local proxy disconnected");
        offset += static_cast<size_t>(received);
    }
}

bool valid_type(uint32_t raw) {
    return raw >= static_cast<uint32_t>(MessageType::WORLD_READY) &&
           raw <= static_cast<uint32_t>(MessageType::FRAME_L3_TO_L4);
}

bool valid_lane(uint32_t raw) { return raw <= static_cast<uint32_t>(Lane::HEALTH); }

}  // namespace

std::vector<uint8_t> encode(const Envelope &envelope) {
    if (envelope.payload.size() > MAX_PAYLOAD_BYTES) throw std::invalid_argument("sidecar payload exceeds maximum");
    std::vector<uint8_t> out;
    out.reserve(HEADER_BYTES + envelope.payload.size());
    out.insert(out.end(), MAGIC, MAGIC + sizeof(MAGIC));
    put_u32(out, PROTOCOL_VERSION);
    put_u32(out, static_cast<uint32_t>(envelope.type));
    put_u32(out, static_cast<uint32_t>(envelope.source_rank));
    put_u32(out, static_cast<uint32_t>(envelope.target_rank));
    put_u64(out, envelope.session_id);
    put_u32(out, static_cast<uint32_t>(envelope.lane));
    put_u32(out, static_cast<uint32_t>(envelope.payload.size()));
    put_u64(out, envelope.sequence);
    out.insert(out.end(), envelope.payload.begin(), envelope.payload.end());
    return out;
}

Envelope decode(const uint8_t *data, size_t size) {
    if (size < HEADER_BYTES) throw std::invalid_argument("sidecar envelope header is truncated");
    if (std::memcmp(data, MAGIC, sizeof(MAGIC)) != 0) throw std::invalid_argument("sidecar envelope magic mismatch");
    if (get_u32(data + 4) != PROTOCOL_VERSION) throw std::invalid_argument("sidecar protocol version mismatch");
    uint32_t raw_type = get_u32(data + 8);
    uint32_t raw_lane = get_u32(data + 28);
    uint32_t payload_size = get_u32(data + 32);
    if (!valid_type(raw_type)) throw std::invalid_argument("sidecar envelope has unknown message type");
    if (!valid_lane(raw_lane)) throw std::invalid_argument("sidecar envelope has unknown lane");
    if (payload_size > MAX_PAYLOAD_BYTES || size != HEADER_BYTES + payload_size) {
        throw std::invalid_argument("sidecar envelope payload length mismatch");
    }
    Envelope result;
    result.type = static_cast<MessageType>(raw_type);
    result.source_rank = static_cast<int32_t>(get_u32(data + 12));
    result.target_rank = static_cast<int32_t>(get_u32(data + 16));
    result.session_id = get_u64(data + 20);
    result.lane = static_cast<Lane>(raw_lane);
    result.sequence = get_u64(data + 36);
    result.payload.assign(data + HEADER_BYTES, data + size);
    return result;
}

void send_local(int fd, const Envelope &envelope) {
    std::vector<uint8_t> bytes = encode(envelope);
    write_all(fd, bytes.data(), bytes.size());
}

Envelope recv_local(int fd) {
    std::vector<uint8_t> bytes(HEADER_BYTES);
    read_all(fd, bytes.data(), bytes.size());
    uint32_t payload_size = get_u32(bytes.data() + 32);
    if (payload_size > MAX_PAYLOAD_BYTES) throw std::invalid_argument("sidecar payload exceeds maximum");
    bytes.resize(HEADER_BYTES + payload_size);
    if (payload_size != 0) read_all(fd, bytes.data() + HEADER_BYTES, payload_size);
    return decode(bytes.data(), bytes.size());
}

}  // namespace simpler::mpi_sidecar
