#!/usr/bin/env python3
"""Write gemm_simt/data for the 64x64x64 bf16 check used on the HBM path.

A = I + E_01, B = I + E_10, so C = AB has
C[0,0] = 2, C[0,1] = 1, C[1,0] = 1, and the other diagonal entries 1.
bf16 1.0 is 0x3f80 and bf16 2.0 is 0x4000.
"""

from pathlib import Path

N = 64
ONE = 0x3F80


def matrix():
    vals = [0] * (N * N)
    for i in range(N):
        vals[i * N + i] = ONE
    return vals


def emit(name, vals):
    body = ",".join(f"0x{v:04x}" for v in vals) + ","
    return f"__global uint16_t {name}[] = {{\n    {body}\n}};\n"


def main():
    a = matrix()
    b = matrix()
    c = [0] * (N * N)
    a[0 * N + 1] = ONE
    b[1 * N + 0] = ONE
    text = (
        emit("A_raw", a)
        + "\n"
        + emit("B_raw", b)
        + "\n"
        + emit("C_raw", c)
        + "\n"
        + f"const uint32_t M_val = {N};\n"
        + f"const uint32_t K_val = {N};\n"
        + f"const uint32_t N_val = {N};\n"
    )
    Path("data").write_text(text)


if __name__ == "__main__":
    main()
