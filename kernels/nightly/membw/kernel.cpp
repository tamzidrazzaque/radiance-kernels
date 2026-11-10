// Memory bandwidth test (see membw.h). Per cluster, each phase streams MEMBW_BYTES per engine from its own buffer, so
// nothing hits in the L2. Warps map round-robin to the cluster's cores (warp w on core w % 2, mu_schedule.cpp).
//   A: Gemmini DMA only: warp 0 lane 0 (core 0) issues mvins of 16 rows x 64 B (each request one 64 B line).
//   B: Gemmini as in A, plus Muon loads (64 B stride, one line per lane) from core 1's warps only, so the producer's
//      MMIO command stores do not queue behind streaming loads on its own core.
//   C: Muon loads from core 1's warps only.
//   D: Muon loads from every warp on both cores.
//   E: Gemmini DMA only, as in A, but on a full L2.
#include <nightly/device.h>
#include <nightly/mx.h>
#include "membw.h"

static constexpr uint32_t NMVIN = MEMBW_BYTES / MEMBW_MVIN_BYTES;
static constexpr uint32_t NLINES = MEMBW_BYTES / MEMBW_LINE;

static void gemmini_stream(uint32_t src) {
  for (uint32_t k = 0; k < NMVIN; k++)   // 64 scratchpad rows per mvin, cycling over rows 0..4095
    mx::cmd(mx::MVIN, mx::host_addr(src + k * MEMBW_MVIN_BYTES), (16ull << 48) | (64ull << 32) | ((k * 64u) % 4096u));
  mx::fence();
}

static uint32_t muon_stream(uint32_t src, uint32_t idx, uint32_t n) {
  uint32_t s = 0;
  for (uint32_t l = idx; l < NLINES; l += n) s += *(volatile const uint32_t *)(src + l * MEMBW_LINE);
  return s;
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  const bool core1 = (warp & 1u) != 0;
  const uint32_t c1_idx = (warp / 2) * MU_NUM_THREADS + lane, c1_n = tpb / 2;   // dense index over core 1's threads
  volatile uint32_t *sink = (volatile uint32_t *)(MEMBW_SINK + (tb * tpb + tid) * 4u);
  uint32_t s = 0;
  if (lead) {
    mx::flush_tlb(); mx::lut_disable(); mx::config_ex(mx::FP8, mx::FP8, mx::BF16); mx::config_ld(MEMBW_LINE, 0);
  } else asm volatile("nop");
  mu_barrier(1, warps);
  // A: Gemmini only
  if (lead) { nightly_phase(MEMBW_A0); gemmini_stream(MEMBW_BUF(0, tb)); nightly_phase(MEMBW_A1); }
  else asm volatile("nop");
  mu_barrier(2, warps);
  // B: Gemmini (core 0 lead) + Muon (core 1)
  if (lead) nightly_phase(MEMBW_B0); else asm volatile("nop");
  if (lead) gemmini_stream(MEMBW_BUF(1, tb));
  else if (core1) s += muon_stream(MEMBW_BUF(2, tb), c1_idx, c1_n);
  else asm volatile("nop");
  *sink = s;   // the store needs every loaded value
  mu_barrier(3, warps);
  if (lead) nightly_phase(MEMBW_B1); else asm volatile("nop");
  // C: Muon, core 1 only
  if (core1) s += muon_stream(MEMBW_BUF(3, tb), c1_idx, c1_n); else asm volatile("nop");
  *sink = s;
  mu_barrier(4, warps);
  if (lead) nightly_phase(MEMBW_C1); else asm volatile("nop");
  // D: Muon, both cores
  s += muon_stream(MEMBW_BUF(4, tb), tid, tpb);
  *sink = s;
  mu_barrier(5, warps);
  if (lead) nightly_phase(MEMBW_D1); else asm volatile("nop");
  // E: Gemmini only again, now on a full L2 (1.25 MiB streamed through a 512 KiB cache): every miss evicts
  if (lead) { gemmini_stream(MEMBW_BUF(5, tb)); nightly_phase(MEMBW_E1); }
  else asm volatile("nop");
  mu_barrier(6, warps);
  nightly_kernel_end(tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
