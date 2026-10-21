#include <stdint.h>
#include <radiance.h>
#ifdef RAD_HOST_TIMING
#include <rad_host.h>
#endif

#include "generated/config.h"
#include <nightly/host.h>
#include "generated/symbols.h"

// Quiet NaNs cannot equal any native STREAM result (all inputs/results are
// finite positive values).  Poisoning the whole NOBITS output before release
// makes a skipped or partial GPU write fail even when the same ELF is rerun
// against backing memory left over from an earlier invocation.
static void poison_output() {
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_OUTPUT_ADDR));
  for (uint32_t i = 0; i < STREAM_ELEMENTS; ++i) {
    output[i] = 0x7FC00000u | (i & 0x003FFFFFu);
  }
}

static uint64_t digest_output() {
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_OUTPUT_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
#if STREAM_READBACK_SAMPLES
  for (uint32_t j = 0; j < STREAM_READBACK_SAMPLES; ++j) {
    const uint32_t i = STREAM_READBACK_SAMPLES == 1 ? 0 :
        static_cast<uint32_t>((static_cast<uint64_t>(j) *
                               (STREAM_ELEMENTS - 1u)) /
                              (STREAM_READBACK_SAMPLES - 1u));
    digest ^= output[i];
    digest *= 0x100000001B3ULL;
  }
#else
  for (uint32_t i = 0; i < STREAM_ELEMENTS; ++i) {
    digest ^= output[i];
    digest *= 0x100000001B3ULL;
  }
#endif
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA5A50000u ^ i) ||
        after[i] != (0x5A5A0000u ^ i)) return false;
  }
  return true;
}

int main() {
  tohost = 0;
  *tocpu = 0;
  // Make launch state explicit.  FireSim targets normally hold this reset at
  // boot, but asserting it here also makes a repeated invocation deterministic.
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  poison_output();
  // Pre-arm every selected cluster's launch mailbox while Muon is held in
  // reset.  On release each core observes the launch word, runs the kernel,
  // fences its L0d after a block-wide barrier, and only then posts END.
  nightly_host_launch(MU_NUM_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");
#ifdef RAD_HOST_TIMING
  const uint64_t release_cycle = rad_rdcycle();
#endif
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);
  const bool completed = nightly_host_wait(MU_NUM_CLUSTERS) != 0;
#ifdef RAD_HOST_TIMING
  const uint64_t completion_cycle = rad_rdcycle();
  if (completed) nightly_host_report(MU_NUM_CLUSTERS);
#endif
  asm volatile("fence rw, rw" ::: "memory");
  const bool passed = completed && guards_intact() &&
      digest_output() == STREAM_EXPECTED_READBACK_DIGEST;
#ifdef RAD_HOST_TIMING
  rad_puts("HOST_RELEASE_TO_DONE_CYCLES=");
  rad_putu(completion_cycle - release_cycle);
  rad_putc('\n');
  rad_flush();
#endif
  tohost = passed ? 1 : 3;
  asm volatile("fence rw, rw" ::: "memory");
  for (;;) {}
}
