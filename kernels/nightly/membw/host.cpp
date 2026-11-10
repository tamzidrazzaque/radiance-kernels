// Memory bandwidth test host: launch, wait, print per-phase bytes per GPU cycle (see membw.h).
#include <nightly/host.h>
#include "membw.h"

static void rate(const char *name, uint64_t bytes, uint32_t cycles) {
  nh_puts(name); nh_puts(" bytes="); nh_putu(bytes); nh_puts(" cycles="); nh_putu(cycles);
  nh_puts(" B/cyc="); nh_putfix(cycles ? bytes * 100 / cycles : 0, 2); nh_puts("\n");
}

int main() {
  nh_puts("membw: "); nh_putu(MEMBW_KB); nh_puts(" KiB per engine per cluster per phase, clusters=");
  nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  uint32_t a = 0, b = 0, c = 0;   // slowest cluster's phase length: the phase is over when every cluster is done
  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++) {
    const uint32_t a0 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_A0), a1 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_A1);
    const uint32_t b0 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_B0), b1 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_B1);
    const uint32_t c1 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_C1);
    const uint32_t d1 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_D1);
    const uint32_t e1 = nightly_pb_read(cl, NIGHTLY_PB_PHASE + MEMBW_E1);
    nh_puts("cluster "); nh_putu(cl); nh_puts(":\n");
    rate("  A gemmini       ", MEMBW_BYTES, a1 - a0);
    rate("  B gemmini+muon1 ", 2ull * MEMBW_BYTES, b1 - b0);
    rate("  C muon core1    ", MEMBW_BYTES, c1 - b1);
    rate("  D muon both     ", MEMBW_BYTES, d1 - c1);
    rate("  E gemmini (full L2)", MEMBW_BYTES, e1 - d1);
    if (a1 - a0 > a) a = a1 - a0;
    if (b1 - b0 > b) b = b1 - b0;
    if (c1 - b1 > c) c = c1 - b1;
  }
  nh_puts("all clusters:\n");
  rate("  A gemmini       ", (uint64_t)NIGHTLY_CLUSTERS * MEMBW_BYTES, a);
  rate("  B gemmini+muon  ", 2ull * NIGHTLY_CLUSTERS * MEMBW_BYTES, b);
  rate("  C muon core1    ", (uint64_t)NIGHTLY_CLUSTERS * MEMBW_BYTES, c);
  nh_puts(ok ? "PASS\n" : "FAIL (timeout)\n"); nh_flush();
  exit(ok ? 0 : 1);
}
