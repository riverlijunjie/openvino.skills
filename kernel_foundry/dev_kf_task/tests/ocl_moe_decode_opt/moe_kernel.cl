
// [USER_INSTRUCTIONS_START]

// [USER_INSTRUCTIONS_END]

// [EVOLVE_START]
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_gate_up(
    const __global int* expert_list,       // [1 x MAX_TOPK] = [1x8]
    const __global uchar* gate_weight_addr,  // [128 x 768 x 2048] in u4 packed
    const __global half* gate_scale_addr,    // [128 x 768 x 16]
    const __global uchar* gate_zp_addr,      // [128 x 768 x 16] in u4 packed
    const __global uchar* up_weight_addr,    // [128 x 768 x 2048] in u4 packed
    const __global half* up_scale_addr,      // [128 x 768 x 16]
    const __global uchar* up_zp_addr,        // [128 x 768 x 16] in u4 packed
    __global half* x,                        // [1 x 2048]
    __global half* y)                        // [8 x 768]
{

}
// [EVOLVE_END]
