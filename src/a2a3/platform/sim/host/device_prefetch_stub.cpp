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

void aicpu_prefetch_init(void *sdma_workspace, uint32_t suppress_window, bool debug_enabled) {
    (void)sdma_workspace;
    (void)suppress_window;
    (void)debug_enabled;
}

void aicpu_prefetch_deinit() {}

bool aicpu_prefetch_reserve_channel(int channel_idx) {
    (void)channel_idx;
    return false;
}

void aicpu_prefetch_issue_reserved(
    void *tensor_addr, size_t tensor_size, void *instr_addr, size_t instr_size, int32_t instr_kernel_id,
    int channel_idx
) {
    (void)tensor_addr;
    (void)tensor_size;
    (void)instr_addr;
    (void)instr_size;
    (void)instr_kernel_id;
    (void)channel_idx;
}

void aicpu_prefetch_tensor(void *addr, size_t size, int channel_idx) {
    (void)addr;
    (void)size;
    (void)channel_idx;
}

bool aicpu_prefetch_available() { return false; }

uint32_t aicpu_prefetch_channel_count() { return 0; }
