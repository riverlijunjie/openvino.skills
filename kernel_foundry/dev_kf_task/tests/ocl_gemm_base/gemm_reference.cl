// [REFERENCE_START]
//
// Simple row-major GEMM: C[M,N] = A[M,K] x B[K,N]
//
// Each work-item computes one element of C.
// Global work size: (N, M)  [id(0)=col, id(1)=row]
//
__kernel void gemm(__global const half* A,
                       __global const half* B,
                       __global half* C,
                       const int M,
                       const int K,
                       const int N)
{
    const int col = get_global_id(0); // output column in [0, N)
    const int row = get_global_id(1); // output row    in [0, M)

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);

    C[row * N + col] = convert_half(acc);
}
// [REFERENCE_END]
