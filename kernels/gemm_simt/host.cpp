#include <rad_host.h>
#include <stdint.h>

// Device ELF symbol C_raw from kernel.radiance.elf (bss, uint16_t[64*64]).
// Fuse places device VA + 0x100000000. This host does not change the GEMM.
#define C_DEV 0x10001064u
#define N 64u

static uint16_t expected_bf16(uint32_t row, uint32_t col) {
  // A = I + E_01, B = I + E_10, so C = AB = I + E_01 + E_10 + E_00.
  // bf16 1.0 = 0x3f80, bf16 2.0 = 0x4000.
  if (row == 0 && col == 0) return 0x4000;
  if (row == 0 && col == 1) return 0x3f80;
  if (row == 1 && col == 0) return 0x3f80;
  if (row == col) return 0x3f80;
  return 0;
}
static uint32_t expected_pair(uint32_t row, uint32_t pair) {
  uint32_t lo = expected_bf16(row, pair * 2u);
  uint32_t hi = expected_bf16(row, pair * 2u + 1u);
  return lo | (hi << 16);
}

int main() {
  rad_puts("Hello from Rocket: gemm_simt 64x64x64 bf16 AB\n");
  rad_flush();

  // GPU is held in reset (defaultReset). Release it onto the device ELF.
  *(volatile uint32_t *)0x41000000u = 1u;
  rad_wait_cycles(1000);
  *(volatile uint32_t *)0x41000000u = 0u;

  volatile uint32_t *C = (volatile uint32_t *)rad_gpu_ptr(C_DEV);
  // Do not touch C until the wait ends, so this load is the first host view
  // of the result rather than a cached zero.
  rad_wait_cycles(400000ull);

  uint32_t wrong = 0, first = 0, got0 = C[0];
  for (uint32_t row = 0; row < N; row++) {
    for (uint32_t pair = 0; pair < N / 2u; pair++) {
      uint32_t idx = row * (N / 2u) + pair;
      uint32_t got = C[idx];
      uint32_t exp = expected_pair(row, pair);
      if (got != exp) {
        if (wrong == 0) first = idx;
        wrong++;
      }
    }
  }
  rad_puts("check: wrong=");
  rad_putu(wrong);
  rad_puts("/2048 first=");
  rad_putu(first);
  rad_puts(" got=");
  rad_putx(wrong ? C[first] : got0);
  rad_puts(wrong ? "\nFAIL\n" : "\nPASS\n");
  rad_flush();
  return wrong ? 1 : 0;
}
