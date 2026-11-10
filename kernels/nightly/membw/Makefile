# Memory bandwidth test: per cluster, Gemmini DMA alone, Gemmini DMA + Muon loads, and Muon loads alone.
#   make [DEFS=-DMEMBW_KB=128]   -> build/{1,2}sm/membw.soc.elf
PROJECT   = membw
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = membw.h
include ../nightly.mk
