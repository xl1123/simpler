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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simpler::mpi_sidecar {

constexpr uint32_t PROTOCOL_VERSION = 2;
constexpr size_t HEADER_BYTES = 44;
constexpr uint32_t MAX_PAYLOAD_BYTES = 16U * 1024U * 1024U;

enum class MessageType : uint32_t {
    WORLD_READY = 1,
    OPEN_SESSION = 2,
    OPEN_SESSION_REPLY = 3,
    FRAME = 4,
    CLOSE_SESSION = 5,
    ERROR = 6,
    SHUTDOWN = 7,
    FRAME_L4_TO_L3 = 8,
    FRAME_L3_TO_L4 = 9,
};

enum class Lane : uint32_t { BOOTSTRAP = 0, COMMAND = 1, HEALTH = 2 };

struct Envelope {
    MessageType type{MessageType::ERROR};
    int32_t source_rank{-1};
    int32_t target_rank{-1};
    uint64_t session_id{0};
    Lane lane{Lane::BOOTSTRAP};
    uint64_t sequence{0};
    std::vector<uint8_t> payload;
};

std::vector<uint8_t> encode(const Envelope &envelope);
Envelope decode(const uint8_t *data, size_t size);
void send_local(int fd, const Envelope &envelope);
Envelope recv_local(int fd);

}  // namespace simpler::mpi_sidecar
