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
#include "host/host_prefetch_setup.h"

#include <acl/acl.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <strings.h>
#include <vector>

#include "common/unified_log.h"

namespace {

struct StarsChannelFlagInfo {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56];
};

struct StarsChannelInfo {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint64_t sq_base;
    uint64_t sq_reg_base;
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4];
};

static_assert(sizeof(StarsChannelFlagInfo) == 64, "Flag info must be 64 bytes");
static_assert(sizeof(StarsChannelInfo) == 64, "Channel info must be 64 bytes");

using RtStreamGetSqidFn = int (*)(const void *stream, uint32_t *sqId);
using RtStreamGetCqidFn = int (*)(const void *stream, uint32_t *cqId, uint32_t *logicCqId);

constexpr size_t SDMA_WORKSPACE_SIZE = 16 * 1024;

std::vector<void *> g_prefetch_streams;
void *g_workspace_device_ptr = nullptr;
int g_cached_device_id = -1;
int g_cached_channel_count = 0;

struct HostPrefetchSetupTimer {
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};
    const char *outcome{"unknown"};
    int channel_count{0};

    ~HostPrefetchSetupTimer() {
        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            std::chrono::steady_clock::now() - start
        );
        LOG_INFO_V0(
            "SDMA prefetch: host_prefetch_setup outcome=%s channels=%d total=%.3fms", outcome, channel_count,
            elapsed.count()
        );
    }
};

bool sdma_prefetch_enabled_by_env() {
    const char *mode = std::getenv("PTO_SDMA_PREFETCH_MODE");
    if (mode != nullptr && *mode != '\0') {
        if (strcasecmp(mode, "baseline") == 0 || std::strcmp(mode, "0") == 0) {
            return false;
        }
        if (strcasecmp(mode, "twoslot") == 0 || std::strcmp(mode, "1") == 0) {
            return false;
        }
        if (strcasecmp(mode, "sdma") == 0 || std::strcmp(mode, "2") == 0) {
            return true;
        }
        if (strcasecmp(mode, "sdma_fake") == 0 || strcasecmp(mode, "fake") == 0 || std::strcmp(mode, "3") == 0) {
            return false;
        }
    }

    const char *value = std::getenv("PTO_ENABLE_SDMA_PREFETCH");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    return !(std::strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0 ||
             strcasecmp(value, "no") == 0);
}

int resolve_channel_count(int requested_count) {
    if (requested_count <= 0) {
        return requested_count;
    }

    const char *env = std::getenv("PTO_SDMA_PREFETCH_CHANNELS");
    if (env == nullptr || *env == '\0') {
        return requested_count;
    }

    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || parsed <= 0) {
        LOG_INFO_V0("SDMA prefetch: ignore invalid PTO_SDMA_PREFETCH_CHANNELS=%s", env);
        return requested_count;
    }

    int override_count = static_cast<int>(parsed);
    int final_count = override_count < requested_count ? override_count : requested_count;
    if (final_count < 1) {
        final_count = 1;
    }
    LOG_INFO_V0(
        "SDMA prefetch: channel override requested=%d env=%d final=%d", requested_count, override_count,
        final_count
    );
    return final_count;
}

}  // namespace

void *host_prefetch_setup(int channel_count) {
    HostPrefetchSetupTimer setup_timer;
    if (!sdma_prefetch_enabled_by_env()) {
        setup_timer.outcome = "disabled_by_env";
        LOG_INFO_V0("SDMA prefetch: disabled by env");
        return nullptr;
    }

    channel_count = resolve_channel_count(channel_count);
    setup_timer.channel_count = channel_count;
    if (channel_count <= 0) {
        setup_timer.outcome = "invalid_channel_count";
        LOG_INFO_V0("SDMA prefetch: disabled (invalid channel_count=%d)", channel_count);
        return nullptr;
    }

    auto rt_stream_get_sqid = reinterpret_cast<RtStreamGetSqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetSqid"));
    auto rt_stream_get_cqid = reinterpret_cast<RtStreamGetCqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetCqid"));
    if (!rt_stream_get_sqid || !rt_stream_get_cqid) {
        setup_timer.outcome = "missing_rt_symbols";
        LOG_INFO_V0("SDMA prefetch: rtStreamGetSqid/Cqid not found, skipping");
        return nullptr;
    }

    int32_t dev_id = -1;
    (void)aclrtGetDevice(&dev_id);
    void *ctx = nullptr;
    (void)aclrtGetCurrentContext(&ctx);
    LOG_INFO_V0("SDMA prefetch: setup start (device=%d ctx=%p channels=%d)", dev_id, ctx, channel_count);

    if (g_workspace_device_ptr != nullptr && g_cached_device_id == dev_id &&
        g_cached_channel_count == channel_count && static_cast<int>(g_prefetch_streams.size()) == channel_count) {
        setup_timer.outcome = "cached_reuse";
        LOG_INFO_V0("SDMA prefetch: reusing cached STARS workspace (device=%d channels=%d)", dev_id, channel_count);
        return g_workspace_device_ptr;
    }

    if (!g_prefetch_streams.empty() || g_workspace_device_ptr != nullptr) {
        host_prefetch_teardown(nullptr);
    }

    void *workspace = nullptr;
    std::vector<StarsChannelInfo> channel_infos(static_cast<size_t>(channel_count));
    g_prefetch_streams.reserve(static_cast<size_t>(channel_count));
    for (int i = 0; i < channel_count; ++i) {
        void *stream = nullptr;
        int rc = aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream *>(&stream), 0, 0x20);
        if (rc != 0 || stream == nullptr) {
            setup_timer.outcome = "stream_create_failed";
            LOG_INFO_V0("SDMA prefetch: create device stream %d/%d failed (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        g_prefetch_streams.push_back(stream);

        StarsChannelInfo &ch = channel_infos[static_cast<size_t>(i)];
        ch.dev_id = static_cast<uint32_t>(dev_id);

        int32_t stream_id = 0;
        (void)aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &stream_id);
        ch.stream_id = static_cast<uint32_t>(stream_id);
        rc = rt_stream_get_sqid(stream, &ch.sq_id);
        if (rc != 0) {
            setup_timer.outcome = "get_sqid_failed";
            LOG_INFO_V0("SDMA prefetch: get sqid failed for stream %d/%d (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        rc = rt_stream_get_cqid(stream, &ch.cq_id, &ch.logic_cq_id);
        if (rc != 0) {
            setup_timer.outcome = "get_cqid_failed";
            LOG_INFO_V0("SDMA prefetch: get cqid failed for stream %d/%d (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        if (i < 4 || i == channel_count - 1) {
            LOG_INFO_V0(
                "SDMA prefetch: stream[%d/%d] created (sid=%d sq=%u cq=%u)", i, channel_count, stream_id,
                ch.sq_id, ch.cq_id
            );
        }
    }

    {
        int rc = aclrtMalloc(&workspace, SDMA_WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != 0) {
            setup_timer.outcome = "workspace_malloc_failed";
            LOG_ERROR("SDMA prefetch: workspace malloc failed");
            goto fail_streams;
        }
        (void)aclrtMemset(workspace, SDMA_WORKSPACE_SIZE, 0, SDMA_WORKSPACE_SIZE);
        LOG_INFO_V0("SDMA prefetch: workspace allocated at %p size=%zu", workspace, SDMA_WORKSPACE_SIZE);

        StarsChannelFlagInfo flag_info = {};
        flag_info.totalQueueNum = static_cast<uint32_t>(channel_infos.size());

        rc = aclrtMemcpy(workspace, sizeof(flag_info), &flag_info, sizeof(flag_info), ACL_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            setup_timer.outcome = "flag_info_copy_failed";
            LOG_ERROR("SDMA prefetch: copy flag info failed (rc=%d)", rc);
            goto fail_workspace;
        }

        size_t channel_infos_size = channel_infos.size() * sizeof(StarsChannelInfo);
        void *channel_info_dev = static_cast<uint8_t *>(workspace) + sizeof(StarsChannelFlagInfo);
        rc = aclrtMemcpy(
            channel_info_dev, channel_infos_size, channel_infos.data(), channel_infos_size, ACL_MEMCPY_HOST_TO_DEVICE
        );
        if (rc != 0) {
            setup_timer.outcome = "channel_info_copy_failed";
            LOG_ERROR("SDMA prefetch: copy channel info failed (rc=%d)", rc);
            goto fail_workspace;
        }
    }

    g_workspace_device_ptr = workspace;
    g_cached_device_id = dev_id;
    g_cached_channel_count = channel_count;
    setup_timer.outcome = "initialized";
    LOG_INFO_V0("SDMA prefetch: STARS channel IDs initialized for AICPU HAL query");
    return workspace;

fail_workspace:
    aclrtFree(workspace);
fail_streams:
    for (void *stream : g_prefetch_streams) {
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    }
    g_prefetch_streams.clear();
    return nullptr;
}

void host_prefetch_teardown(void *workspace) {
    void *workspace_to_free = workspace != nullptr ? workspace : g_workspace_device_ptr;
    if (workspace_to_free) {
        aclrtFree(workspace_to_free);
    }
    g_workspace_device_ptr = nullptr;
    g_cached_device_id = -1;
    g_cached_channel_count = 0;
    for (void *stream : g_prefetch_streams) {
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    }
    g_prefetch_streams.clear();
}
