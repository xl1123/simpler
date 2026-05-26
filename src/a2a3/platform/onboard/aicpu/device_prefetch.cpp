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
#include "aicpu/device_prefetch.h"

#include <cinttypes>
#include <cstring>
#include <dlfcn.h>

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "common/platform_config.h"
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

struct StarsSdmaCmoSqe {
    uint8_t type : 6;
    uint8_t l1_lock : 1;
    uint8_t l1_unlock : 1;

    uint8_t ie : 2;
    uint8_t pre_p : 2;
    uint8_t post_p : 2;
    uint8_t wr_cqe : 1;
    uint8_t reserved_hdr : 1;

    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t d2d_offset_flag : 1;
    uint32_t res6 : 3;

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    uint32_t src_offset_low;
    uint32_t dst_offset_low;
    uint16_t src_offset_high;
    uint16_t dst_offset_high;
    uint32_t res_last[1];
};

static_assert(sizeof(StarsSdmaCmoSqe) == 64, "CMO SQE must be 64 bytes");
static_assert(sizeof(StarsChannelFlagInfo) == 64, "Flag info must be 64 bytes");
static_assert(sizeof(StarsChannelInfo) == 64, "Channel info must be 64 bytes");

constexpr uint8_t STARS_SQE_TYPE_SDMA = 11;
constexpr uint8_t CMO_OPCODE_PREFETCH = 6;
constexpr uint8_t DEFAULT_KERNEL_CREDIT = 254;
constexpr int PREFETCH_MAX_KERNEL_ID = 32;
constexpr uint32_t SQCQ_QUERY_INFO_LENGTH = 8;

using drvError_t = int32_t;

enum drvSqCqType_t : uint32_t {
    DRV_NORMAL_TYPE = 0,
    DRV_CALLBACK_TYPE = 1,
    DRV_LOGIC_TYPE = 2,
    DRV_SHM_TYPE = 3,
};

enum drvSqCqPropType_t : uint32_t {
    DRV_SQCQ_PROP_SQ_STATUS = 0,
    DRV_SQCQ_PROP_SQ_HEAD = 1,
    DRV_SQCQ_PROP_SQ_TAIL = 2,
    DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE = 3,
    DRV_SQCQ_PROP_SQ_CQE_STATUS = 4,
    DRV_SQCQ_PROP_SQ_REG_BASE = 5,
    DRV_SQCQ_PROP_SQ_BASE = 6,
    DRV_SQCQ_PROP_SQ_DEPTH = 7,
};

struct halSqCqQueryInfo {
    drvSqCqType_t type;
    uint32_t tsId;
    uint32_t sqId;
    uint32_t cqId;
    drvSqCqPropType_t prop;
    uint32_t value[SQCQ_QUERY_INFO_LENGTH];
};

bool g_prefetch_enabled = false;
volatile StarsChannelInfo *g_channel_info = nullptr;
uint32_t g_channel_count = 0;
uint32_t g_prefetch_suppress_window = 0;
volatile uint32_t g_prefetch_submit_lock[PLATFORM_MAX_CORES] = {0};
volatile uint32_t g_prefetch_skip_queue_full[PLATFORM_MAX_CORES] = {0};
volatile uint32_t g_prefetch_channel_suppress_remaining[PLATFORM_MAX_CORES] = {0};
volatile uint32_t g_prefetch_skip_suppressed[PLATFORM_MAX_CORES] = {0};
volatile uint64_t g_prefetch_last_instr_addr[PLATFORM_MAX_CORES] = {0};
volatile uint32_t g_prefetch_last_instr_size[PLATFORM_MAX_CORES] = {0};
volatile uint32_t g_prefetch_skip_duplicate_instr[PLATFORM_MAX_CORES] = {0};
volatile uint8_t g_prefetch_instr_kernel_seen[PREFETCH_MAX_KERNEL_ID] = {0};
volatile uint32_t g_prefetch_skip_duplicate_instr_kernel = 0;
volatile uint32_t g_prefetch_attempt_count = 0;
volatile uint32_t g_prefetch_issue_count = 0;
volatile uint64_t g_prefetch_attempt_bytes = 0;
volatile uint64_t g_prefetch_issue_bytes = 0;
volatile uint64_t g_prefetch_attempt_cycles = 0;
volatile uint64_t g_prefetch_issue_cycles = 0;
bool g_prefetch_debug_enabled = false;

using HalSqCqQueryFn = drvError_t (*)(uint32_t devId, halSqCqQueryInfo *info);

HalSqCqQueryFn g_hal_sq_cq_query = nullptr;
bool g_hal_sq_cq_query_resolved = false;

void resolve_hal_sq_cq_query() {
    if (g_hal_sq_cq_query_resolved) {
        return;
    }
    g_hal_sq_cq_query = reinterpret_cast<HalSqCqQueryFn>(dlsym(RTLD_DEFAULT, "halSqCqQuery"));
    if (g_hal_sq_cq_query == nullptr) {
        LOG_ERROR("SDMA prefetch: failed to resolve halSqCqQuery: %s", dlerror());
    }
    g_hal_sq_cq_query_resolved = true;
}

uint64_t query_value_to_u64(const uint32_t value[SQCQ_QUERY_INFO_LENGTH]) {
    return static_cast<uint64_t>(value[0]) | (static_cast<uint64_t>(value[1]) << 32);
}

uint64_t query_value_to_u64_reversed(const uint32_t value[SQCQ_QUERY_INFO_LENGTH]) {
    return static_cast<uint64_t>(value[1]) | (static_cast<uint64_t>(value[0]) << 32);
}

bool query_sqcq_u64(
    uint32_t dev_id, uint32_t ts_id, drvSqCqType_t type, uint32_t sq_id, uint32_t cq_id,
    drvSqCqPropType_t prop, uint64_t *value
) {
    halSqCqQueryInfo query = {};
    query.type = type;
    query.tsId = ts_id;
    query.sqId = sq_id;
    query.cqId = cq_id;
    query.prop = prop;
    drvError_t rc = g_hal_sq_cq_query(dev_id, &query);
    if (rc != 0) {
        return false;
    }
    if (prop == DRV_SQCQ_PROP_SQ_REG_BASE) {
        *value = query_value_to_u64_reversed(query.value);
    } else {
        *value = query_value_to_u64(query.value);
    }
    return true;
}

bool populate_channel_with_hal(volatile StarsChannelInfo *ch, uint32_t channel_idx, uint32_t channel_count) {
    resolve_hal_sq_cq_query();
    if (g_hal_sq_cq_query == nullptr) {
        return false;
    }

    static constexpr uint32_t ts_ids[] = {0, 1};
    for (uint32_t ts_id : ts_ids) {
        uint64_t sq_base = 0;
        uint64_t sq_reg_base = 0;
        uint64_t sq_depth = 0;
        bool ok = query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_BASE, &sq_base
                  ) &&
                  query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_REG_BASE,
                      &sq_reg_base
                  ) &&
                  query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_DEPTH,
                      &sq_depth
                  );
        if (!ok || sq_base == 0 || sq_reg_base == 0 || sq_depth == 0) {
            continue;
        }

        ch->sq_base = sq_base;
        ch->sq_reg_base = sq_reg_base;
        ch->sq_depth = static_cast<uint32_t>(sq_depth);
        if (channel_idx < 4 || channel_idx + 1 == channel_count) {
            LOG_INFO_V0(
                "SDMA prefetch: HAL channel[%u] sid=%u sq=%u cq=%u dev=%u ts=%u type=%d base=0x%llx reg=0x%llx depth=%u",
                channel_idx, ch->stream_id, ch->sq_id, ch->cq_id, ch->dev_id, ts_id,
                static_cast<int>(DRV_NORMAL_TYPE), static_cast<unsigned long long>(ch->sq_base),
                static_cast<unsigned long long>(ch->sq_reg_base), ch->sq_depth
            );
        }
        return true;
    }

    for (uint32_t ts_id : ts_ids) {
        uint64_t sq_base = 0;
        uint64_t sq_reg_base = 0;
        uint64_t sq_depth = 0;
        bool ok = query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_BASE, &sq_base
                  ) &&
                  query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_REG_BASE,
                      &sq_reg_base
                  ) &&
                  query_sqcq_u64(
                      ch->dev_id, ts_id, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_DEPTH, &sq_depth
                  );
        if (ok && sq_base != 0 && sq_reg_base != 0 && sq_depth != 0) {
            LOG_INFO_V0(
                "SDMA prefetch: HAL found only SHM queue for channel[%u] sid=%u sq=%u cq=%u dev=%u ts=%u; NORMAL path unresolved",
                channel_idx, ch->stream_id, ch->sq_id, ch->cq_id, ch->dev_id, ts_id
            );
            break;
        }
    }

    LOG_INFO_V0(
        "SDMA prefetch: HAL query failed for channel[%u] sid=%u sq=%u cq=%u dev=%u", channel_idx,
        ch->stream_id, ch->sq_id, ch->cq_id, ch->dev_id
    );
    return false;
}

inline void prefetch_lock(int channel_idx) {
    while (__atomic_test_and_set(&g_prefetch_submit_lock[channel_idx], __ATOMIC_ACQUIRE)) {
    }
}

inline void prefetch_unlock(int channel_idx) { __atomic_clear(&g_prefetch_submit_lock[channel_idx], __ATOMIC_RELEASE); }

inline uint32_t consume_prefetch_suppress_window(int channel_idx) {
    while (true) {
        uint32_t remaining = __atomic_load_n(&g_prefetch_channel_suppress_remaining[channel_idx], __ATOMIC_RELAXED);
        if (remaining == 0) {
            return 0;
        }
        uint32_t desired = remaining - 1;
        if (__atomic_compare_exchange_n(
                &g_prefetch_channel_suppress_remaining[channel_idx], &remaining, desired, false,
                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED
            )) {
            return remaining;
        }
    }
}

bool prepare_prefetch_channel(int *channel_idx, bool count_attempt, bool consume_suppress) {
    if (!g_prefetch_enabled || g_channel_info == nullptr || g_channel_count == 0 || channel_idx == nullptr ||
        *channel_idx < 0) {
        return false;
    }

    *channel_idx %= static_cast<int>(g_channel_count);

    if (count_attempt && g_prefetch_debug_enabled) {
        __atomic_add_fetch(&g_prefetch_attempt_count, 1, __ATOMIC_RELAXED);
    }

    if (consume_suppress) {
        uint32_t suppress_remaining = consume_prefetch_suppress_window(*channel_idx);
        if (suppress_remaining != 0) {
            uint32_t skipped = g_prefetch_debug_enabled
                                   ? __atomic_add_fetch(
                                         &g_prefetch_skip_suppressed[*channel_idx], 1, __ATOMIC_RELAXED
                                     )
                                   : 0;
            if (g_prefetch_debug_enabled && (skipped <= 4 || (skipped % 64) == 0)) {
                LOG_INFO_V0(
                    "SDMA prefetch: suppressed, skip issue (channel=%d skipped=%u remaining_after=%u)",
                    *channel_idx, skipped, suppress_remaining - 1
                );
            }
            return false;
        }
    }

    return true;
}

void fill_cmo_prefetch_sqe(
    volatile StarsSdmaCmoSqe *sqe, uint64_t src_addr, uint32_t length, uint16_t stream_id, uint16_t task_id
) {
    std::memset(const_cast<StarsSdmaCmoSqe *>(sqe), 0, sizeof(StarsSdmaCmoSqe));

    sqe->type = STARS_SQE_TYPE_SDMA;
    sqe->rt_streamid = stream_id;
    sqe->task_id = task_id;
    sqe->kernel_credit = DEFAULT_KERNEL_CREDIT;

    sqe->opcode = CMO_OPCODE_PREFETCH;
    sqe->length = length;

    sqe->src_addr_low = static_cast<uint32_t>(src_addr & 0xFFFFFFFFu);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFFu);

    sqe->qos = 6;
    sqe->partid = 63;
    sqe->sssv = 1;
    sqe->dssv = 1;
    sqe->sns = 1;
    sqe->dns = 1;
}

}  // namespace

void aicpu_prefetch_init(void *sdma_workspace, uint32_t suppress_window, bool debug_enabled) {
    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    g_prefetch_suppress_window = suppress_window;
    g_prefetch_debug_enabled = debug_enabled;
    g_prefetch_attempt_count = 0;
    g_prefetch_issue_count = 0;
    g_prefetch_attempt_bytes = 0;
    g_prefetch_issue_bytes = 0;
    g_prefetch_attempt_cycles = 0;
    g_prefetch_issue_cycles = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        g_prefetch_channel_suppress_remaining[i] = 0;
        g_prefetch_skip_suppressed[i] = 0;
        g_prefetch_last_instr_addr[i] = 0;
        g_prefetch_last_instr_size[i] = 0;
        g_prefetch_skip_duplicate_instr[i] = 0;
    }
    for (int i = 0; i < PREFETCH_MAX_KERNEL_ID; ++i) {
        g_prefetch_instr_kernel_seen[i] = 0;
    }
    g_prefetch_skip_duplicate_instr_kernel = 0;

    if (sdma_workspace == nullptr) {
        LOG_INFO_V0("SDMA prefetch: disabled (no workspace provided)");
        return;
    }

    uint8_t *base = static_cast<uint8_t *>(sdma_workspace);
    auto *flag_info = reinterpret_cast<volatile StarsChannelFlagInfo *>(base);
    g_channel_info = reinterpret_cast<volatile StarsChannelInfo *>(base + sizeof(StarsChannelFlagInfo));
    if (g_prefetch_debug_enabled) {
        LOG_INFO_V0(
            "SDMA prefetch: workspace=%p flag_info=%p channel_info=%p flag=0x%x totalQueueNum=%u",
            sdma_workspace, const_cast<StarsChannelFlagInfo *>(flag_info),
            const_cast<StarsChannelInfo *>(g_channel_info), flag_info->flag, flag_info->totalQueueNum
        );
    }
    g_channel_count = flag_info->totalQueueNum;
    if (g_channel_count == 0 || g_channel_count > PLATFORM_MAX_CORES) {
        if (g_prefetch_debug_enabled) {
            LOG_INFO_V0("SDMA prefetch: invalid channel count %u", g_channel_count);
        }
        g_channel_info = nullptr;
        g_channel_count = 0;
        return;
    }

    for (uint32_t i = 0; i < g_channel_count; ++i) {
        volatile StarsChannelInfo *ch = g_channel_info + i;
        if ((ch->sq_base == 0 || ch->sq_reg_base == 0 || ch->sq_depth == 0) &&
            !populate_channel_with_hal(ch, i, g_channel_count)) {
            g_channel_info = nullptr;
            g_channel_count = 0;
            return;
        }
    }

    if (g_channel_info->sq_base == 0 || g_channel_info->sq_reg_base == 0 || g_channel_info->sq_depth == 0) {
        if (g_prefetch_debug_enabled) {
            LOG_INFO_V0(
                "SDMA prefetch: disabled (channel info not initialized after HAL query: sq_base=%llu sq_reg=%llu depth=%u)",
                static_cast<unsigned long long>(g_channel_info->sq_base),
                static_cast<unsigned long long>(g_channel_info->sq_reg_base), g_channel_info->sq_depth
            );
        }
        g_channel_info = nullptr;
        return;
    }

    g_prefetch_enabled = true;
    if (g_prefetch_debug_enabled) {
        for (uint32_t i = 0; i < g_channel_count && i < 4; ++i) {
            volatile StarsChannelInfo *ch = g_channel_info + i;
            LOG_INFO_V0(
                "SDMA prefetch: channel[%u] sq_head=%u sq_tail=%u sq_base=0x%llx sq_reg=0x%llx depth=%u sq_id=%u cq_id=%u stream=%u",
                i, ch->sq_head, ch->sq_tail, static_cast<unsigned long long>(ch->sq_base),
                static_cast<unsigned long long>(ch->sq_reg_base), ch->sq_depth, ch->sq_id, ch->cq_id,
                ch->stream_id
            );
        }
        LOG_INFO_V0(
            "SDMA prefetch: enabled (channels=%u sq_base=0x%llx sq_reg=0x%llx depth=%u stream=%u)",
            g_channel_count, static_cast<unsigned long long>(g_channel_info->sq_base),
            static_cast<unsigned long long>(g_channel_info->sq_reg_base), g_channel_info->sq_depth,
            g_channel_info->stream_id
        );
    }
}

void aicpu_prefetch_deinit() {
    uint32_t attempt_count = __atomic_load_n(&g_prefetch_attempt_count, __ATOMIC_ACQUIRE);
    uint32_t issue_count = __atomic_load_n(&g_prefetch_issue_count, __ATOMIC_ACQUIRE);
    uint64_t attempt_bytes = __atomic_load_n(&g_prefetch_attempt_bytes, __ATOMIC_ACQUIRE);
    uint64_t issue_bytes = __atomic_load_n(&g_prefetch_issue_bytes, __ATOMIC_ACQUIRE);
    uint64_t attempt_cycles = __atomic_load_n(&g_prefetch_attempt_cycles, __ATOMIC_ACQUIRE);
    uint64_t issue_cycles = __atomic_load_n(&g_prefetch_issue_cycles, __ATOMIC_ACQUIRE);
    uint64_t suppressed_count = 0;
    uint64_t queue_full_count = 0;
    uint64_t duplicate_instr_count = 0;
    uint64_t duplicate_instr_kernel_count = __atomic_load_n(&g_prefetch_skip_duplicate_instr_kernel, __ATOMIC_ACQUIRE);
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        suppressed_count += __atomic_load_n(&g_prefetch_skip_suppressed[i], __ATOMIC_ACQUIRE);
        queue_full_count += __atomic_load_n(&g_prefetch_skip_queue_full[i], __ATOMIC_ACQUIRE);
        duplicate_instr_count += __atomic_load_n(&g_prefetch_skip_duplicate_instr[i], __ATOMIC_ACQUIRE);
    }
    if (g_prefetch_debug_enabled) {
        LOG_INFO_V0(
            "SDMA prefetch issue summary: enabled=%d attempts=%u issues=%u bytes=%" PRIu64
            " issue_bytes=%" PRIu64 " suppressed=%" PRIu64 " queue_full=%" PRIu64
            " dup_instr=%" PRIu64 " dup_instr_kernel=%" PRIu64 " total=%.3fus avg=%.3fus"
            " issue_total=%.3fus issue_avg=%.3fus",
            g_prefetch_enabled ? 1 : 0, attempt_count, issue_count, attempt_bytes, issue_bytes, suppressed_count,
            queue_full_count, duplicate_instr_count, duplicate_instr_kernel_count, cycles_to_us(attempt_cycles),
            attempt_count > 0 ? cycles_to_us(attempt_cycles) / attempt_count : 0.0, cycles_to_us(issue_cycles),
            issue_count > 0 ? cycles_to_us(issue_cycles) / issue_count : 0.0
        );
    }

    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        __atomic_store_n(&g_prefetch_submit_lock[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_queue_full[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_channel_suppress_remaining[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_suppressed[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_last_instr_addr[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_last_instr_size[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_duplicate_instr[i], 0, __ATOMIC_RELEASE);
    }
    for (int i = 0; i < PREFETCH_MAX_KERNEL_ID; ++i) {
        __atomic_store_n(&g_prefetch_instr_kernel_seen[i], 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_prefetch_skip_duplicate_instr_kernel, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_bytes, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_bytes, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_cycles, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_cycles, 0, __ATOMIC_RELEASE);
}

bool aicpu_prefetch_reserve_channel(int channel_idx) { return prepare_prefetch_channel(&channel_idx, true, true); }

void aicpu_prefetch_issue_reserved(
    void *tensor_addr, size_t tensor_size, void *instr_addr, size_t instr_size, int32_t instr_kernel_id,
    int channel_idx
) {
    uint64_t start_cycle = g_prefetch_debug_enabled ? get_sys_cnt_aicpu() : 0;
    bool issued = false;

    do {
        if (!g_prefetch_enabled || tensor_addr == nullptr || tensor_size == 0) {
            break;
        }

        if (!prepare_prefetch_channel(&channel_idx, false, false)) {
            break;
        }

        volatile StarsChannelInfo *ch = g_channel_info + channel_idx;
        prefetch_lock(channel_idx);

        uint32_t sq_head = __atomic_load_n(&ch->sq_head, __ATOMIC_ACQUIRE);
        uint32_t sq_tail = __atomic_load_n(&ch->sq_tail, __ATOMIC_ACQUIRE);
        uint32_t sq_depth = ch->sq_depth;
        if (sq_depth == 0) {
            LOG_INFO_V0("SDMA prefetch: channel %d has zero depth", channel_idx);
            prefetch_unlock(channel_idx);
            break;
        }

        uint32_t new_tail = (sq_tail + 1) % sq_depth;
        if (new_tail == sq_head) {
            uint32_t skipped = g_prefetch_debug_enabled
                                   ? __atomic_add_fetch(
                                         &g_prefetch_skip_queue_full[channel_idx], 1, __ATOMIC_RELAXED
                                     )
                                   : 0;
            if (g_prefetch_debug_enabled && (skipped <= 4 || (skipped % 64) == 0)) {
                LOG_INFO_V0(
                    "SDMA prefetch: queue full, skip issue (channel=%d head=%u tail=%u depth=%u skipped=%u)",
                    channel_idx, sq_head, sq_tail, sq_depth, skipped
                );
            }
            prefetch_unlock(channel_idx);
            break;
        }

        const bool want_instr = (instr_addr != nullptr && instr_size > 0 && sq_depth > 1);
        bool duplicate_instr = false;
        bool duplicate_instr_kernel = false;
        if (want_instr) {
            if (instr_kernel_id >= 0 && instr_kernel_id < PREFETCH_MAX_KERNEL_ID) {
                duplicate_instr_kernel =
                    (__atomic_load_n(&g_prefetch_instr_kernel_seen[instr_kernel_id], __ATOMIC_RELAXED) != 0);
                if (duplicate_instr_kernel && g_prefetch_debug_enabled) {
                    __atomic_add_fetch(&g_prefetch_skip_duplicate_instr_kernel, 1, __ATOMIC_RELAXED);
                }
            }
            duplicate_instr =
                (__atomic_load_n(&g_prefetch_last_instr_addr[channel_idx], __ATOMIC_RELAXED) ==
                     reinterpret_cast<uint64_t>(instr_addr) &&
                 __atomic_load_n(&g_prefetch_last_instr_size[channel_idx], __ATOMIC_RELAXED) ==
                     static_cast<uint32_t>(instr_size));
            if (duplicate_instr && g_prefetch_debug_enabled) {
                __atomic_add_fetch(&g_prefetch_skip_duplicate_instr[channel_idx], 1, __ATOMIC_RELAXED);
            }
        }
        const uint32_t tail_after_tensor = new_tail;
        const uint32_t tail_after_instr = (tail_after_tensor + 1u) % sq_depth;
        const bool issue_two =
            want_instr && !duplicate_instr && !duplicate_instr_kernel && (tail_after_instr != sq_head);
        const uint32_t final_tail = issue_two ? tail_after_instr : tail_after_tensor;

        if (g_prefetch_debug_enabled) {
            __atomic_add_fetch(&g_prefetch_attempt_bytes, static_cast<uint64_t>(tensor_size), __ATOMIC_RELAXED);
            if (issue_two) {
                __atomic_add_fetch(&g_prefetch_attempt_bytes, static_cast<uint64_t>(instr_size), __ATOMIC_RELAXED);
            }
        }

        volatile StarsSdmaCmoSqe *sq_base = reinterpret_cast<volatile StarsSdmaCmoSqe *>(ch->sq_base);

        uint16_t task_id0 = static_cast<uint16_t>((sq_tail + sq_depth - sq_head) % sq_depth);
        fill_cmo_prefetch_sqe(
            sq_base + sq_tail, reinterpret_cast<uint64_t>(tensor_addr), static_cast<uint32_t>(tensor_size),
            static_cast<uint16_t>(ch->stream_id), task_id0
        );
        uint32_t issue_idx =
            g_prefetch_debug_enabled ? __atomic_add_fetch(&g_prefetch_issue_count, 1, __ATOMIC_RELAXED) : 0;
        if (g_prefetch_debug_enabled) {
            __atomic_add_fetch(&g_prefetch_issue_bytes, static_cast<uint64_t>(tensor_size), __ATOMIC_RELAXED);
        }
        if (g_prefetch_debug_enabled && issue_idx <= 8) {
            LOG_INFO_V0(
                "SDMA prefetch: issue[%u] channel=%d tensor addr=0x%llx size=%zu sq_head=%u sq_tail=%u new_tail=%u depth=%u",
                issue_idx, channel_idx, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(tensor_addr)),
                tensor_size, sq_head, sq_tail, new_tail, sq_depth
            );
        }

        if (issue_two) {
            uint16_t task_id1 = static_cast<uint16_t>((tail_after_tensor + sq_depth - sq_head) % sq_depth);
            fill_cmo_prefetch_sqe(
                sq_base + tail_after_tensor, reinterpret_cast<uint64_t>(instr_addr), static_cast<uint32_t>(instr_size),
                static_cast<uint16_t>(ch->stream_id), task_id1
            );
            if (g_prefetch_debug_enabled) {
                issue_idx = __atomic_add_fetch(&g_prefetch_issue_count, 1, __ATOMIC_RELAXED);
                __atomic_add_fetch(&g_prefetch_issue_bytes, static_cast<uint64_t>(instr_size), __ATOMIC_RELAXED);
            }
            if (g_prefetch_debug_enabled && issue_idx <= 8) {
                LOG_INFO_V0(
                    "SDMA prefetch: issue[%u] channel=%d instr addr=0x%llx size=%zu head=%u tail=%u final_tail=%u depth=%u",
                    issue_idx, channel_idx, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(instr_addr)),
                    instr_size, sq_head, sq_tail, final_tail, sq_depth
                );
            }
            __atomic_store_n(
                &g_prefetch_last_instr_addr[channel_idx], reinterpret_cast<uint64_t>(instr_addr), __ATOMIC_RELEASE
            );
            __atomic_store_n(&g_prefetch_last_instr_size[channel_idx], static_cast<uint32_t>(instr_size), __ATOMIC_RELEASE);
            if (instr_kernel_id >= 0 && instr_kernel_id < PREFETCH_MAX_KERNEL_ID) {
                __atomic_store_n(&g_prefetch_instr_kernel_seen[instr_kernel_id], 1, __ATOMIC_RELEASE);
            }
        } else if (duplicate_instr && g_prefetch_debug_enabled) {
            uint32_t dup_count = __atomic_load_n(&g_prefetch_skip_duplicate_instr[channel_idx], __ATOMIC_RELAXED);
            if (dup_count <= 4 || (dup_count % 64) == 0) {
                LOG_INFO_V0(
                    "SDMA prefetch: skip duplicate instr sqe (channel=%d addr=0x%llx size=%zu dup_count=%u)",
                    channel_idx, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(instr_addr)), instr_size,
                    dup_count
                );
            }
        } else if (duplicate_instr_kernel && g_prefetch_debug_enabled) {
            uint32_t dup_count = __atomic_load_n(&g_prefetch_skip_duplicate_instr_kernel, __ATOMIC_RELAXED);
            if (dup_count <= 4 || (dup_count % 64) == 0) {
                LOG_INFO_V0(
                    "SDMA prefetch: skip duplicate instr kernel (kid=%d addr=0x%llx size=%zu dup_kernel_count=%u)",
                    instr_kernel_id, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(instr_addr)),
                    instr_size, dup_count
                );
            }
        }

        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&ch->sq_tail, final_tail, __ATOMIC_RELEASE);
        __atomic_store_n(
            &g_prefetch_channel_suppress_remaining[channel_idx], g_prefetch_suppress_window, __ATOMIC_RELEASE
        );

        volatile uint32_t *doorbell = reinterpret_cast<volatile uint32_t *>(ch->sq_reg_base + 8);
        *doorbell = final_tail;

        prefetch_unlock(channel_idx);
        issued = true;
    } while (false);

    if (g_prefetch_debug_enabled) {
        uint64_t elapsed = get_sys_cnt_aicpu() - start_cycle;
        __atomic_add_fetch(&g_prefetch_attempt_cycles, elapsed, __ATOMIC_RELAXED);
        if (issued) {
            __atomic_add_fetch(&g_prefetch_issue_cycles, elapsed, __ATOMIC_RELAXED);
        }
    }
}

void aicpu_prefetch_tensor(void *addr, size_t size, int channel_idx) {
    if (!aicpu_prefetch_reserve_channel(channel_idx)) {
        return;
    }
    aicpu_prefetch_issue_reserved(addr, size, nullptr, 0, -1, channel_idx);
}

bool aicpu_prefetch_available() { return g_prefetch_enabled; }

uint32_t aicpu_prefetch_channel_count() { return g_channel_count; }
