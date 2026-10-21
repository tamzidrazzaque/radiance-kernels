#include "generated/config.h"

#include <stdint.h>

#define STREAM_WARPS 4
#define NIGHTLY_OCC STREAM_WARPS
#define NIGHTLY_CLUSTERS MU_NUM_CLUSTERS
#include <nightly/device.h>

extern "C" {
extern __global float stream_a[];
extern __global float stream_b[];
extern __global float stream_c[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  const uint32_t global_tid = block_id * threads_per_block + tid;
  const uint32_t global_threads = MU_NUM_CLUSTERS * threads_per_block;
  for (uint32_t i = global_tid; i < STREAM_ELEMENTS;) {
#if STREAM_KIND == 0
    stream_c[i] = stream_a[i];
#elif STREAM_KIND == 1
    stream_b[i] = 2.0f * stream_c[i];
#elif STREAM_KIND == 2
    stream_c[i] = stream_a[i] + stream_b[i];
#elif STREAM_KIND == 3
    stream_a[i] = stream_b[i] + 2.0f * stream_c[i];
#endif
    if (STREAM_ELEMENTS - i <= global_threads) break;
    i += global_threads;
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() {
  return nightly_main(kernel_body, nullptr);
}
