// Shared by kernel and host: sizes and buffer addresses (device / GPU-local addresses).
#pragma once
#ifndef MEMBW_KB
#define MEMBW_KB 128          // bytes each engine moves per cluster per phase, in KiB
#endif
#define MEMBW_BYTES (MEMBW_KB * 1024u)
#define MEMBW_LINE 64u        // bytes per DRAM row = one cache line
#define MEMBW_MVIN_BYTES (16u * MEMBW_LINE)   // one mvin: 16 rows x 64 fp8 columns
// one 16 MiB buffer per (stream, cluster); never initialized, only read
#define MEMBW_BUF(stream, cl) (0x30000000u + ((stream) * 4u + (cl)) * 0x01000000u)
#define MEMBW_SINK 0x3F000000u
// phase stamps (printBuf PHASE slots, cluster core 0): A = Gemmini, B = Gemmini + Muon (core 1), C = Muon (core 1),
// D = Muon (both cores)
#define MEMBW_A0 0
#define MEMBW_A1 1
#define MEMBW_B0 2
#define MEMBW_B1 3
#define MEMBW_C1 4
#define MEMBW_D1 5
#define MEMBW_E1 6
