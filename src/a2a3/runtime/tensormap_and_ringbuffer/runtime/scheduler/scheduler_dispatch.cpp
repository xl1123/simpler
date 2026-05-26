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
#include "scheduler_context.h"

#include <algorithm>
#include <cinttypes>

#include "common.h"  // debug_assert

#include "common/unified_log.h"
#include "aicpu/device_prefetch.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/l2_perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// =============================================================================
// Dispatch helpers
// =============================================================================

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;

struct PrefetchFeatureTier {
    bool enable_instr;
    uint32_t instr_cap_bytes;
    uint32_t min_suppress_window;
};
}

const char *SchedulerContext::shape_name(PTO2ResourceShape shape) {
    switch (shape) {
    case PTO2ResourceShape::AIC:
        return "AIC";
    case PTO2ResourceShape::AIV:
        return "AIV";
    case PTO2ResourceShape::MIX:
        return "MIX";
    case PTO2ResourceShape::DUMMY:
        return "DUMMY";
    }
    return "UNKNOWN";
}

bool SchedulerContext::has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const {
    // Cross-thread read of peer trackers without explicit synchronization. The
    // backing `core_states_` is a naturally aligned uint64_t; aarch64 guarantees
    // single-copy atomicity for an 8-byte aligned load, so no torn read. The
    // value is consumed only as a scheduling *hint* — a stale read at worst
    // causes one missed/extra pending dispatch, corrected on the next iteration.
    // Drain-mode cross-thread writes are serialized by handle_drain_mode's ack
    // barrier (all peers spin out of the dispatch path before any tracker
    // mutation), so this routine is never racing the drain worker.
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        if (t == self_thread_idx) continue;
        if (core_trackers_[t].get_idle_core_offset_states(shape).has_value()) {
            return true;
        }
    }
    return false;
}

int SchedulerContext::pop_ready_tasks_batch(
    PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
) {
#if PTO2_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#if PTO2_SCHED_PROFILING
    extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
    uint64_t t_pop_start = get_sys_cnt_aicpu();
    int count = sched_->get_ready_tasks_batch(
        shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx],
        l2_perf.local_dispatch_count
    );
    l2_perf.sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
#else
    int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
    // pop_hit / pop_miss are PTO2_PROFILING-gated (not the inner verbose tier)
    // so the v2 JSON dispatch records carry queue-health stats on default builds.
    if (count > 0) {
        l2_perf.pop_hit += count;
    } else {
        l2_perf.pop_miss++;
    }
#else
    (void)thread_idx;
    int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
    return count;
}

uint64_t SchedulerContext::get_prefetch_logical_span(const PTO2TaskPayload &payload) {
    if (payload.prefetch_issue_bytes == 0) {
        return 0;
    }
    return payload.prefetch_filter_bytes / payload.prefetch_issue_bytes;
}

static PrefetchFeatureTier classify_prefetch_features(const PTO2TaskPayload &payload) {
    static constexpr uint64_t kSmallIssueBytes = 20 * 1024;
    static constexpr uint64_t kSmallFilterBytes = 384 * 1024;
    static constexpr uint64_t kInstrEnableIssueBytes = 32 * 1024;
    static constexpr uint64_t kMidSpanThreshold = 16;
    static constexpr uint64_t kWideSpanThreshold = 48;

    const uint64_t issue_bytes = payload.prefetch_issue_bytes;
    const uint64_t filter_bytes = payload.prefetch_filter_bytes;
    const uint64_t logical_span = issue_bytes == 0 ? 0 : filter_bytes / issue_bytes;
    const bool structured_block_task = (payload.tensor_count == 4) &&
                                       (payload.scalar_count == 2 || payload.scalar_count == 4 ||
                                        payload.scalar_count == 6);

    if (!structured_block_task) {
        return {false, 0u, 31u};
    }
    if (issue_bytes < kSmallIssueBytes || filter_bytes < kSmallFilterBytes) {
        return {false, 0u, 31u};
    }
    if (logical_span > kWideSpanThreshold) {
        return {false, 0u, 31u};
    }
    if (logical_span > kMidSpanThreshold) {
        return {false, 0u, 15u};
    }
    if (issue_bytes >= kInstrEnableIssueBytes) {
        return {true, 512u, 11u};
    }
    return {false, 0u, 23u};
}

uint32_t SchedulerContext::get_scheduler_prefetch_suppress_window(const PTO2TaskSlotState &slot_state) const {
    if (slot_state.payload == nullptr) {
        return prefetch_suppress_window_;
    }
    const PrefetchFeatureTier tier = classify_prefetch_features(*slot_state.payload);
    return prefetch_suppress_window_ >= tier.min_suppress_window ? prefetch_suppress_window_ :
                                                                  tier.min_suppress_window;
}

int SchedulerContext::get_scheduler_prefetch_channel_idx(int channel_idx) {
    if (channel_idx < 0) {
        return channel_idx;
    }
    uint32_t channel_count = aicpu_prefetch_channel_count();
    if (channel_count == 0) {
        return channel_idx;
    }
    return channel_idx % static_cast<int>(channel_count);
}

bool SchedulerContext::should_attempt_task_prefetch(const PTO2TaskSlotState &slot_state, int channel_idx) const {
    if (prefetch_debug_enabled_) {
        prefetch_considered_count_.fetch_add(1, std::memory_order_relaxed);
    }
    if (prefetch_mode_ != Runtime::PREFETCH_MODE_SDMA) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_not_sdma_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    if (!aicpu_prefetch_available()) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_not_available_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    if (slot_state.payload == nullptr) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_null_payload_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    if (!slot_state.active_mask.subtask_active(PTO2SubtaskSlot::AIC)) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_no_valid_tensor_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    const PTO2TaskPayload &payload = *slot_state.payload;
    if (payload.prefetch_addr == 0 || payload.prefetch_issue_bytes == 0 || payload.prefetch_filter_bytes == 0) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_no_valid_tensor_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    if (payload.prefetch_filter_bytes < prefetch_min_bytes_) {
        if (prefetch_debug_enabled_) {
            prefetch_skip_below_min_bytes_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    int suppress_channel_idx = get_scheduler_prefetch_channel_idx(channel_idx);
    uint32_t scheduler_suppress_window = get_scheduler_prefetch_suppress_window(slot_state);
    if (scheduler_suppress_window > 0 && suppress_channel_idx >= 0 && suppress_channel_idx < RUNTIME_MAX_WORKER) {
        while (true) {
            uint32_t remaining =
                prefetch_scheduler_suppress_remaining_[suppress_channel_idx].load(std::memory_order_relaxed);
            if (remaining == 0) {
                break;
            }
            uint32_t desired = remaining - 1;
            if (prefetch_scheduler_suppress_remaining_[suppress_channel_idx].compare_exchange_weak(
                    remaining, desired, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                if (prefetch_debug_enabled_) {
                    prefetch_skip_scheduler_suppressed_.fetch_add(1, std::memory_order_relaxed);
                }
                return false;
            }
        }
    }
    return true;
}

void SchedulerContext::issue_task_prefetch(const PTO2TaskSlotState &slot_state, int channel_idx) const {
    uint64_t start_cycle = prefetch_debug_enabled_ ? get_sys_cnt_aicpu() : 0;
    bool eligible_task = false;
    do {
        PTO2TaskPayload *payload = slot_state.payload;
        if (payload == nullptr) {
            break;
        }
        const PTO2TaskPayload &p = *payload;
        const PrefetchFeatureTier feature_tier = classify_prefetch_features(p);
        eligible_task = true;
        if (!aicpu_prefetch_reserve_channel(channel_idx)) {
            break;
        }

        void *instr_ptr = nullptr;
        size_t instr_size = 0;
        int32_t instr_kernel_id = INVALID_KERNEL_ID;
        if (slot_state.active_mask.subtask_active(PTO2SubtaskSlot::AIC) && slot_state.task != nullptr) {
            int32_t kid = slot_state.task->kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)];
            if (kid != INVALID_KERNEL_ID) {
                if (!feature_tier.enable_instr) {
                    if (prefetch_debug_enabled_) {
                        prefetch_skip_instr_feature_disabled_.fetch_add(1, std::memory_order_relaxed);
                    }
                    payload->instr_prefetch_addr = 0;
                } else if (kid >= 0 && kid < RUNTIME_MAX_FUNC_ID &&
                           prefetch_instr_kernel_seen_[kid].load(std::memory_order_relaxed) != 0) {
                    if (prefetch_debug_enabled_) {
                        prefetch_skip_instr_kernel_scheduler_.fetch_add(1, std::memory_order_relaxed);
                    }
                    payload->instr_prefetch_addr = 0;
                } else {
                    uint64_t callable_addr = get_function_bin_addr(kid);
                    if (callable_addr != 0) {
                        const auto *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
                        const uint64_t resolved = callable->resolved_addr();
                        if (resolved != 0) {
                            const uint32_t bin = callable->binary_size();
                            if (bin > 0) {
                                const uint32_t cap = feature_tier.instr_cap_bytes;
                                const uint32_t n = (bin < cap) ? bin : cap;
                                payload->instr_prefetch_addr = resolved;
                                instr_kernel_id = kid;
                                instr_ptr = reinterpret_cast<void *>(resolved);
                                instr_size = static_cast<size_t>(n);
                            } else {
                                payload->instr_prefetch_addr = 0;
                            }
                        } else {
                            payload->instr_prefetch_addr = 0;
                        }
                    } else {
                        payload->instr_prefetch_addr = 0;
                    }
                }
            } else {
                payload->instr_prefetch_addr = 0;
            }
        } else {
            payload->instr_prefetch_addr = 0;
        }

        if (prefetch_debug_enabled_) {
            prefetch_task_count_.fetch_add(1, std::memory_order_relaxed);
            prefetch_tensor_count_.fetch_add(1, std::memory_order_relaxed);
            prefetch_total_bytes_.fetch_add(p.prefetch_issue_bytes, std::memory_order_relaxed);
            if (instr_ptr != nullptr) {
                prefetch_total_bytes_.fetch_add(static_cast<uint64_t>(instr_size), std::memory_order_relaxed);
            }
        }
        aicpu_prefetch_issue_reserved(
            reinterpret_cast<void *>(p.prefetch_addr), static_cast<size_t>(p.prefetch_issue_bytes), instr_ptr,
            instr_size, instr_kernel_id, channel_idx
        );
        if (instr_kernel_id >= 0 && instr_kernel_id < RUNTIME_MAX_FUNC_ID && instr_ptr != nullptr) {
            prefetch_instr_kernel_seen_[instr_kernel_id].store(1, std::memory_order_relaxed);
        }
    } while (false);

    if (prefetch_debug_enabled_) {
        uint64_t elapsed = get_sys_cnt_aicpu() - start_cycle;
        prefetch_control_cycles_.fetch_add(elapsed, std::memory_order_relaxed);
        if (eligible_task) {
            prefetch_eligible_control_cycles_.fetch_add(elapsed, std::memory_order_relaxed);
        }
    }
}

PTO2TaskSlotState *SchedulerContext::select_next_task_prefetch_target(
    PTO2TaskSlotState *const *batch, int got, int bi
) {
    for (int next = bi + 1; next < got; ++next) {
        if (batch[next] != nullptr) {
            return batch[next];
        }
    }
    return nullptr;
}

void SchedulerContext::maybe_prefetch_next_task(
    int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, CoreTracker &tracker,
    PTO2TaskSlotState *const *batch, int got, int bi, CoreTracker::BitStates candidate_cores
) {
    (void)thread_idx;
    if (phase != CoreTracker::DispatchPhase::IDLE) {
        return;
    }
    PTO2TaskSlotState *prefetch_target = select_next_task_prefetch_target(batch, got, bi);
    if (prefetch_target == nullptr) {
        return;
    }
    if (!candidate_cores.has_value()) {
        candidate_cores = tracker.get_dispatchable_cores(shape, phase);
    }
    if (!candidate_cores.has_value()) {
        return;
    }
    int core_offset = candidate_cores.pop_first();
    if (core_offset < 0) {
        return;
    }
    int prefetch_core_id = tracker.get_core_id_by_offset(core_offset);
    if (prefetch_core_id < 0 || !should_attempt_task_prefetch(*prefetch_target, prefetch_core_id)) {
        return;
    }
    issue_task_prefetch(*prefetch_target, prefetch_core_id);
    int suppress_channel_idx = get_scheduler_prefetch_channel_idx(prefetch_core_id);
    uint32_t scheduler_suppress_window = get_scheduler_prefetch_suppress_window(*prefetch_target);
    if (scheduler_suppress_window > 0 && suppress_channel_idx >= 0 && suppress_channel_idx < RUNTIME_MAX_WORKER) {
        prefetch_scheduler_suppress_remaining_[suppress_channel_idx].store(
            scheduler_suppress_window, std::memory_order_release
        );
    }
}

void SchedulerContext::build_payload(
    PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    const AsyncCtx &async_ctx, int32_t block_idx
) {
    int32_t slot_idx = static_cast<int32_t>(subslot);
    uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    dispatch_payload.function_bin_addr = callable->resolved_addr();
    auto &payload = *slot_state.payload;
    int n = 0;
    for (int32_t i = 0; i < payload.tensor_count; i++) {
        dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
    }
    for (int32_t i = 0; i < payload.scalar_count; i++) {
        dispatch_payload.args[n++] = payload.scalars[i];
    }
    dispatch_payload.local_context.block_idx = block_idx;
    dispatch_payload.local_context.block_num = slot_state.logical_block_num;
    dispatch_payload.local_context.async_ctx = async_ctx;
    dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
    dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
}

void SchedulerContext::dispatch_subtask_to_core(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, bool to_pending,
    int32_t block_idx
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto core_id = tracker.get_core_id_by_offset(core_offset);
    CoreExecState &core_exec_state = core_exec_states_[core_id];
    core_exec_state.dispatch_seq++;
    uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
    static_assert(
        (TASK_ID_MASK - AICORE_EXIT_SIGNAL + 1) % 2 == 0, "Sentinel skip must be even to preserve dual-buffer parity"
    );
    if (reg_task_id >= AICORE_EXIT_SIGNAL) {
        core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
        reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
    }

    uint32_t buf_idx = reg_task_id & 1u;
    PTO2DispatchPayload &payload = payload_per_core_[core_id][buf_idx];
    DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][buf_idx];
    deferred_slab->count = 0;
    deferred_slab->error_code = PTO2_ERROR_NONE;
    AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_slab);
    build_payload(payload, slot_state, subslot, async_ctx, block_idx);

    if (to_pending) {
        core_exec_state.pending_subslot = subslot;
        core_exec_state.pending_slot_state = &slot_state;
        core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
        if (l2_perf_level_ >= L2PerfLevel::AICPU_TIMING) {
            core_exec_state.pending_dispatch_timestamp = get_sys_cnt_aicpu();
        }
#endif
    } else {
        core_exec_state.running_subslot = subslot;
        core_exec_state.running_slot_state = &slot_state;
        core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
        if (l2_perf_level_ >= L2PerfLevel::AICPU_TIMING) {
            core_exec_state.running_dispatch_timestamp = get_sys_cnt_aicpu();
        }
#endif
        tracker.change_core_state(core_offset);
    }

    LOG_DEBUG(
        "Thread %d: Dispatched %s %s task %" PRId64 " kernel_id=[%d,%d,%d] block_idx=%d/total_blocks=%d to"
        " core_offset=%d core_id=%d reg_task_id=%u",
        thread_idx, to_pending ? "pending" : "idle", subslot_name(subslot),
        static_cast<int64_t>(slot_state.task->task_id.raw), slot_state.task->kernel_id[0],
        slot_state.task->kernel_id[1], slot_state.task->kernel_id[2], block_idx, slot_state.logical_block_num,
        core_offset, core_id, reg_task_id
    );

    write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));
    tracker.set_pending_occupied(core_offset);
}

void SchedulerContext::dispatch_mix_block_to_cluster(
    int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending, int32_t block_idx
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    uint8_t cmask = slot_state.active_mask.core_mask();
    if (cmask & PTO2_SUBTASK_MASK_AIC) {
        bool aic_to_pending = to_pending && !tracker.is_aic_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC, aic_to_pending,
            block_idx
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV0) {
        bool aiv0_to_pending = to_pending && !tracker.is_aiv0_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            thread_idx, tracker.get_aiv0_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV0,
            aiv0_to_pending, block_idx
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV1) {
        bool aiv1_to_pending = to_pending && !tracker.is_aiv1_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            thread_idx, tracker.get_aiv1_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV1,
            aiv1_to_pending, block_idx
        );
    }
}

void SchedulerContext::dispatch_block(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending,
    int32_t block_idx
) {
#if PTO2_PROFILING
    if (is_dump_tensor_enabled()) {
        dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
            thread_idx, slot_state, TensorDumpStage::BEFORE_DISPATCH,
            [](ActiveMask active_mask, int raw_subtask_id) {
                return active_mask.subtask_active(static_cast<PTO2SubtaskSlot>(raw_subtask_id));
            },
            [this](int32_t func_id) {
                return get_function_bin_addr(func_id);
            }
        );
    }
#endif
    if (shape == PTO2ResourceShape::MIX) {
        dispatch_mix_block_to_cluster(thread_idx, core_offset, slot_state, to_pending, block_idx);
    } else if (shape == PTO2ResourceShape::AIC) {
        dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
    } else {
        dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
    }
#if PTO2_PROFILING
    sched_l2_perf_[thread_idx].phase_dispatch_count += __builtin_popcount(slot_state.active_mask.core_mask());
#endif
}

void SchedulerContext::dispatch_shape(
    int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf,
    CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
) {
#if PTO2_SCHED_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#endif
    if (entered_drain) return;

    bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
    auto cores = tracker.get_dispatchable_cores(shape, phase);
    if (!cores.has_value()) return;

    while (cores.has_value() && !entered_drain) {
        int want = cores.count();
        PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
        int got = pop_ready_tasks_batch(shape, thread_idx, local_buf, batch, want);
        if (got == 0) break;

        bool dispatched_any = false;
        for (int bi = 0; bi < got; bi++) {
            PTO2TaskSlotState *slot_state = batch[bi];

            if (slot_state->active_mask.requires_sync_start()) {
                if (is_pending) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
                int32_t available = cores.count();
                if (available < slot_state->logical_block_num) {
                    if (!enter_drain_mode(slot_state, slot_state->logical_block_num)) {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    }
                    for (int rem = bi + 1; rem < got; rem++) {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                    }
                    entered_drain = true;
                    break;
                }
            }

            if (!cores.has_value()) {
                sched_->ready_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                break;
            }

            dispatched_any = true;
            try_pushed = true;
#if PTO2_SCHED_PROFILING
            uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
            // Claim a contiguous range of blocks, hand the slot back to the
            // ready queue immediately, then perform the expensive dispatches.
            // This lets other schedulers concurrently claim and dispatch the
            // remaining blocks of the same SPMD task instead of spinning while
            // this thread fills all its own cores.  Only local `start + b` is
            // read after the push -- `next_block_idx` may already be advanced
            // by another scheduler that popped the slot.
            int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
            int32_t claim = std::min(cores.count(), remaining);
            int32_t start = slot_state->next_block_idx;
            slot_state->next_block_idx += claim;

            if (slot_state->next_block_idx < slot_state->logical_block_num) {
                sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
            }

            for (int32_t b = 0; b < claim; b++) {
                auto core_offset = cores.pop_first();
                dispatch_block(thread_idx, core_offset, *slot_state, shape, is_pending, start + b);
                if (start + b == 0) {
                    maybe_prefetch_next_task(thread_idx, shape, phase, tracker, batch, got, bi, cores);
                }
            }
            made_progress = true;
#if PTO2_SCHED_PROFILING
            l2_perf.sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
        }

        if (!dispatched_any) break;

        if (!cores.has_value()) {
            cores = tracker.get_dispatchable_cores(shape, phase);
        }
    }
}

void SchedulerContext::dispatch_ready_tasks(
    int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES],
    bool pmu_active, bool &made_progress, bool &try_pushed
) {
    using Phase = CoreTracker::DispatchPhase;
    constexpr int32_t MIX_I = static_cast<int32_t>(PTO2ResourceShape::MIX);

    // MIX is handled explicitly at the top of each stage; only AIC/AIV cycle
    // through this 2-elem array, with order toggled by thread parity for
    // shape-level load balancing across threads.
    static constexpr PTO2ResourceShape kAicAivOrder[2][2] = {
        {PTO2ResourceShape::AIC, PTO2ResourceShape::AIV},
        {PTO2ResourceShape::AIV, PTO2ResourceShape::AIC},
    };
    const PTO2ResourceShape *aic_aiv = kAicAivOrder[thread_idx & 1];

#if PTO2_SCHED_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#endif

    // Note: flush_local_bufs is invoked multiple times per pass (mid-function
    // flush + RAII tail flush). local_overflow_count accumulates each batch
    // separately — each entry is counted exactly once (count is zeroed after
    // push_batch). The total reflects "entries this pass pushed to the global
    // queue", which is slightly larger than the pre-refactor "buf residual at
    // pass end" semantics — comparing PTO2_SCHED_PROFILING traces across
    // commits, expect the post-refactor number to be greater-or-equal.
    auto flush_local_bufs = [&]() {
        for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
            auto &lb = local_bufs[s];
#if PTO2_SCHED_PROFILING
            l2_perf.local_overflow_count += lb.count;
#endif
            if (lb.count > 0) {
                sched_->ready_queues[s].push_batch(lb.slot_states, lb.count);
                lb.count = 0;
            }
        }
    };
    // Every return path below must flush; wrap in RAII so we cannot forget.
    // The mid-function flush between IDLE and PENDING is still called
    // explicitly — guard only covers exit.
    struct FlushGuard {
        decltype(flush_local_bufs) &flush_fn;
        ~FlushGuard() { flush_fn(); }
    } flush_guard{flush_local_bufs};

    bool entered_drain = false;

    // ===== IDLE stage =====
    dispatch_shape(
        thread_idx, PTO2ResourceShape::MIX, Phase::IDLE, local_bufs[MIX_I], tracker, entered_drain, made_progress,
        try_pushed
    );
    if (entered_drain) return;

    // MIX-IDLE residual: AIC/AIV (both IDLE and PENDING) yield for this pass.
    // MIX-PENDING below still runs — that is the core of "mix strict priority":
    // pending slots are spent on mix before AIC/AIV get any chance.
    bool skip_aic_aiv = has_residual_mix(local_bufs[MIX_I]);

    if (!skip_aic_aiv) {
        for (int i = 0; i < 2; i++) {
            PTO2ResourceShape s = aic_aiv[i];
            dispatch_shape(
                thread_idx, s, Phase::IDLE, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress,
                try_pushed
            );
            if (entered_drain) return;
        }
    }

    // Flush between IDLE and PENDING so PENDING-stage queue-size checks and any
    // peer-thread reads see the IDLE-stage release_fanin output.
    flush_local_bufs();

    if (pmu_active) return;

    // ===== PENDING stage =====
    // MIX-PENDING gate: skip when a peer has an idle MIX-capable cluster — that
    // peer's next IDLE-MIX iteration will pull the mix task from the global
    // queue (already flushed above) at lower latency than us pre-loading a
    // pending slot here. Forward progress for MIX is preserved: at least one
    // thread will run MIX-IDLE next pass and consume the residual.
    //
    // The gate is NOT subject to skip_aic_aiv — residual mix continues to drain
    // via pending slots on this thread when no peer is idle.
    if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX)) {
        dispatch_shape(
            thread_idx, PTO2ResourceShape::MIX, Phase::PENDING, local_bufs[MIX_I], tracker, entered_drain,
            made_progress, try_pushed
        );
        if (entered_drain) return;
    }

    // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
    // it set; otherwise, escalate iff PENDING-MIX left residual.
    if (!skip_aic_aiv && has_residual_mix(local_bufs[MIX_I])) {
        skip_aic_aiv = true;
    }

    // PENDING-MIX may have re-populated AIC/AIV local_bufs via release_fanin
    // during in-flight completions; flush_guard ensures these don't carry
    // across to the next iteration's IDLE stage.
    if (skip_aic_aiv) return;

    // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
    // will pull from the global queue on its next IDLE pass.
    for (int i = 0; i < 2; i++) {
        PTO2ResourceShape s = aic_aiv[i];
        if (has_idle_in_other_threads(thread_idx, s)) continue;
        dispatch_shape(
            thread_idx, s, Phase::PENDING, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress,
            try_pushed
        );
        if (entered_drain) return;
    }
}

// =============================================================================
// Main scheduler dispatch loop
// =============================================================================

int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {
    always_assert(sched_ != nullptr);
    CoreTracker &tracker = core_trackers_[thread_idx];
    LOG_INFO_V0("Thread %d: resolve_and_dispatch entry", thread_idx);

    PTO2SharedMemoryHeader *header = sched_->sm_header;
    if (!header) {
        LOG_ERROR("PTO2 dispatch: header is null");
        return -1;
    }
    LOG_INFO_V0(
        "Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu", thread_idx, static_cast<void *>(header),
        static_cast<uint64_t>(header->rings[0].task_descriptors_offset),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    Handshake *hank = static_cast<Handshake *>(runtime->workers);
    LOG_INFO_V0(
        "Thread %d: hank=%p, window_size=%lu", thread_idx, static_cast<void *>(hank),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        LOG_INFO_V0("Thread %d: doing one-time init", thread_idx);

#if PTO2_PROFILING
        if (is_dump_tensor_enabled()) {
            dump_tensor_init(orch_to_sched_ ? aicpu_thread_num_ : sched_thread_num_);
        }
#endif

#if PTO2_PROFILING
        // Initialize PMU: program events, start counters, and pop initial buffers
        if (is_pmu_enabled()) {
            pmu_aicpu_init(physical_core_ids_, cores_total_num_);
            LOG_INFO_V0("PMU profiling started on %d cores", cores_total_num_);
        }
#endif

        LOG_INFO_V0("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    LOG_INFO_V0("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_trackers_[thread_idx].core_num());
    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if PTO2_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
    l2_perf.reset();
    l2_perf.l2_perf_enabled = (l2_perf_level_ != L2PerfLevel::DISABLED);
#endif

    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);
    }
    PTO2TaskSlotState *deferred_release_slot_states[PTO2_DEFERRED_RELEASE_CAP];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

    // PMU runs require single-issue dispatch — overlapping in-flight tasks
    // pollute per-task PMU counters, so skip the PENDING pre-load phase.
    // Cached at function scope: is_pmu_enabled() is extern "C" and the
    // compiler cannot hoist it across the dispatch loop on its own.
    const bool pmu_active = is_pmu_enabled();

#if PTO2_PROFILING
    l2_perf.sched_start_ts = get_sys_cnt_aicpu();
#endif

    while (true) {
        if (completed_.load(std::memory_order_acquire)) {
            break;
        }
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        l2_perf.sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (!tracker.has_any_running_cores()) {
            LoopAction action = handle_orchestrator_exit(thread_idx, header, runtime, task_count);
            if (action == LoopAction::BREAK_LOOP) break;
        }

        if (!cores_released && orch_to_sched_) {
            LoopAction action = handle_core_transition(cores_released);
            if (action == LoopAction::BREAK_LOOP) break;
        }

#if PTO2_PROFILING
        CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
#endif

        // Phase 1: Check running cores for completion
        int32_t completed_this_turn = 0;

        bool try_completed = tracker.has_any_running_cores();
        if (try_completed) {
            check_running_cores_for_completion(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
            );
        }
        if (completed_this_turn > 0) {
#if PTO2_SCHED_PROFILING
            sched_->tasks_completed.fetch_add(completed_this_turn, std::memory_order_relaxed);
#endif
            int32_t prev = completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            int32_t new_total = prev + completed_this_turn;
            last_progress_count = new_total;
            if (thread_idx == 0 && task_count > 0) {
                if (new_total <= PROGRESS_VERBOSE_THRESHOLD ||
                    new_total / PROGRESS_LOG_INTERVAL != prev / PROGRESS_LOG_INTERVAL || new_total >= task_count) {
                    LOG_INFO_V9(
                        "PTO2 progress: completed=%d total=%d (%.1f%%)", new_total, task_count,
                        100.0 * new_total / task_count
                    );
                }
            }
        }

        if (rt_ != nullptr && rt_->aicore_mailbox != nullptr &&
            (sched_->async_wait_list.count > 0 || sched_->async_wait_list.pending_completion_count > 0)) {
            AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(
                rt_->aicore_mailbox, sched_, local_bufs, deferred_release_slot_states, deferred_release_count,
                PTO2_DEFERRED_RELEASE_CAP
#if PTO2_SCHED_PROFILING
                ,
                thread_idx
#endif
            );
            if (poll_result.error_code != PTO2_ERROR_NONE) {
                int32_t expected = PTO2_ERROR_NONE;
                header->sched_error_code.compare_exchange_strong(
                    expected, poll_result.error_code, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                break;
            }
            if (poll_result.completed > 0) {
#if PTO2_SCHED_PROFILING
                sched_->tasks_completed.fetch_add(poll_result.completed, std::memory_order_relaxed);
#endif
                int32_t prev = completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                int32_t new_total = prev + poll_result.completed;
                last_progress_count = new_total;
                made_progress = true;
            }
        }

#if PTO2_PROFILING
        if (!try_completed) {
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_perf.sched_complete_cycle);
            if (l2_perf_level_ >= L2PerfLevel::SCHED_PHASES && l2_perf.phase_complete_count > 0) {
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, l2_perf.sched_loop_count,
                    l2_perf.phase_complete_count
                );
                _t0_phase = _t1;
                l2_perf.phase_complete_count = 0;
            }
        }
#endif

        bool try_pushed = false;

        // Phase 2 drain check
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(thread_idx);
            continue;
        }

        // Phase 3: Drain wiring queue (thread 0 only)
        if (thread_idx == 0) {
            int wired = sched_->drain_wiring_queue(orchestrator_done_);
            if (wired > 0) {
                made_progress = true;
#if PTO2_SCHED_PROFILING
                l2_perf.phase_wiring_count += wired;
#endif
            }
        }
#if PTO2_PROFILING
        CYCLE_COUNT_LAP(l2_perf.sched_wiring_cycle);
#endif

        // Phase 3b: Drain dummy ready queue (thread 0 only).
        //
        // Dependency-only tasks bypass AICore dispatch: they go through the
        // scheduler so fanin/fanout edges stay consistent, but completion is
        // signalled inline here. Pinned to thread 0 to avoid cross-thread
        // races and to keep cache hot near the wiring drain above.
        if (thread_idx == 0) {
            constexpr int DUMMY_DRAIN_BATCH = 16;
            PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
            int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
            for (int di = 0; di < dummy_got; di++) {
                PTO2TaskSlotState &dummy_slot = *dummy_batch[di];
#if PTO2_SCHED_PROFILING
                sched_->on_mixed_task_complete(dummy_slot, thread_idx, local_bufs);
#else
                sched_->on_mixed_task_complete(dummy_slot, local_bufs);
#endif
                // Dummy tasks have no subtasks to retire and no fanout pre-conditions
                // beyond their own producers; release self-reference so the slot can
                // reach CONSUMED once all consumers drain.
                deferred_release_slot_states[deferred_release_count++] = &dummy_slot;
                if (deferred_release_count >= PTO2_DEFERRED_RELEASE_CAP) {
                    while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                        (void)sched_->on_task_release(
                            *deferred_release_slot_states[--deferred_release_count], thread_idx
                        );
#else
                        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                    }
                }
                int32_t prev = completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                last_progress_count = prev + 1;
                cur_thread_completed++;
            }
            if (dummy_got > 0) {
                made_progress = true;
            }
        }

        // Phase 4: MIX-strict-priority dispatch with phase-split and
        // cross-thread idle gating. See dispatch_ready_tasks for the policy.
        dispatch_ready_tasks(thread_idx, tracker, local_bufs, pmu_active, made_progress, try_pushed);

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_perf.sched_dispatch_cycle);
            if (l2_perf_level_ >= L2PerfLevel::SCHED_PHASES && l2_perf.phase_dispatch_count > 0) {
                // Per-emit pop deltas via snapshot diff; the cumulative
                // pop_hit / pop_miss stay intact for the cold-path log.
                uint64_t pop_hit_delta = l2_perf.pop_hit - l2_perf.pop_hit_at_last_emit;
                uint64_t pop_miss_delta = l2_perf.pop_miss - l2_perf.pop_miss_at_last_emit;
                // AicpuPhaseRecord's extras are uint32 — a delta that overflows means
                // an emit was missed for ~4 billion pops, which is well outside any
                // realistic dispatch cadence and silently truncates without this guard.
                debug_assert(pop_hit_delta < (1ULL << 32));
                debug_assert(pop_miss_delta < (1ULL << 32));
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, l2_perf.sched_loop_count,
                    l2_perf.phase_dispatch_count, static_cast<uint32_t>(pop_hit_delta),
                    static_cast<uint32_t>(pop_miss_delta)
                );
                _t0_phase = _t1;
                l2_perf.phase_dispatch_count = 0;
                l2_perf.pop_hit_at_last_emit = l2_perf.pop_hit;
                l2_perf.pop_miss_at_last_emit = l2_perf.pop_miss;
            }
        }
#endif

#if !PTO2_PROFILING
        (void)try_completed;
        (void)try_pushed;
#endif

        if (made_progress) {
            idle_iterations = 0;
        } else {
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
            idle_iterations++;

            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                LoopAction action = check_idle_fatal_error(thread_idx, header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (idle_iterations % STALL_LOG_INTERVAL == 0) {
                log_stall_diagnostics(thread_idx, total_tasks_, idle_iterations, last_progress_count);
            }
            if (idle_iterations >= MAX_IDLE_ITERATIONS) {
                return handle_timeout_exit(
                    thread_idx, header, runtime, idle_iterations
#if PTO2_PROFILING
                    ,
                    l2_perf.sched_start_ts
#endif
                );
            } else {
                SPIN_WAIT_HINT();
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
            if (l2_perf_level_ >= L2PerfLevel::SCHED_PHASES) {
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_IDLE_WAIT, _t0_phase, _t1, l2_perf.sched_loop_count, 0
                );
                _t0_phase = _t1;
            }
#endif
        }
    }

    // Drain any entries left in the deferred-release batch. The in-loop flush
    // only fires on idle iterations and on buffer-full; a loop exit while the
    // last iteration made progress can leave entries un-released. Drop them
    // here so every consumed producer slot completes its on_task_release
    // regardless of which loop-exit path fired.
    while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
        (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
    }

#if PTO2_PROFILING
    // Final-drain: emit any pop_hit / pop_miss accrued since the last
    // dispatch emit (typically the trailing idle loops while waiting for
    // orchestrator_done_) as a zero-duration synthetic dispatch record so
    // sum(record.pop_*) reconciles with the run-cumulative counter.
    // Gate on SCHED_PHASES — at lower levels the phase buffer is never
    // flushed (see below), so writing this record would be wasted work.
    if (l2_perf_level_ >= L2PerfLevel::SCHED_PHASES) {
        uint64_t final_pop_hit_delta = l2_perf.pop_hit - l2_perf.pop_hit_at_last_emit;
        uint64_t final_pop_miss_delta = l2_perf.pop_miss - l2_perf.pop_miss_at_last_emit;
        debug_assert(final_pop_hit_delta < (1ULL << 32));
        debug_assert(final_pop_miss_delta < (1ULL << 32));
        if (final_pop_hit_delta != 0 || final_pop_miss_delta != 0) {
            uint64_t t_now = get_sys_cnt_aicpu();
            l2_perf_aicpu_record_phase(
                thread_idx, AicpuPhaseId::SCHED_DISPATCH, t_now, t_now, l2_perf.sched_loop_count, 0,
                static_cast<uint32_t>(final_pop_hit_delta), static_cast<uint32_t>(final_pop_miss_delta)
            );
            l2_perf.pop_hit_at_last_emit = l2_perf.pop_hit;
            l2_perf.pop_miss_at_last_emit = l2_perf.pop_miss;
        }
    }
    log_l2_perf_summary(thread_idx, cur_thread_completed);
#endif

#if PTO2_PROFILING
    if (l2_perf.l2_perf_enabled) {
        l2_perf_aicpu_flush_buffers(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
        if (l2_perf_level_ >= L2PerfLevel::SCHED_PHASES) {
            l2_perf_aicpu_flush_phase_buffers(thread_idx);
        }
    }
#endif
#if PTO2_PROFILING
    if (is_dump_tensor_enabled()) {
        dump_tensor_flush(thread_idx);
    }
#endif
#if PTO2_PROFILING
    if (is_pmu_enabled()) {
        pmu_aicpu_flush_buffers(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
    }
#endif

    return cur_thread_completed;
}
