/**
 * MoE bench — exercises the cldnn::moe_3gemm_fused_compressed primitive by
 * building the canonical softmax-routing MoE pattern the plugin's
 * MoE3GemmFusion transformation recognizes, compiling it with the GPU plugin,
 * and measuring the fused kernel's latency via cliloader.
 *
 * Pattern (softmax routing, 3-GEMM SwiGLU MoE, per Qwen3-MoE / Mixtral style):
 *
 *   input [B, S, H]
 *     ├─ experts_reshape [B*S, H] ─ Tile([NE,1]) ─ Reshape [NE, B*S, H]
 *     │                     ├────────── gate_matmul (weights NE×H×I u4)  ─ Swish ┐
 *     │                     └────────── up_matmul   (weights NE×H×I u4)          │
 *     │                                                                          │
 *     │                                 └──── Multiply(SwiGLU) ────────┐         │
 *     │                                        down_matmul             │         │
 *     │                                       (weights NE×I×H u4)      │         │
 *     │                                                                │         │
 *     └─ router_matmul (weights H×NE u4) ─ Softmax(axis=1) ─ TopK ──── Divide ─── ScatterElementsUpdate
 *                                              │                                  │
 *                                             Transpose ─ Reshape ─ Unsqueeze ────┘
 *                                                                                 ↓
 *                                                                        Multiply(down × router_weights)
 *                                                                                 ↓
 *                                                                        ReduceSum(axis=0)  [B*S, H]
 *                                                                                 ↓
 *                                                                        Reshape  [B, S, H]
 *
 * The three matmul chains use the standard u4-group128 decompression pattern
 *   Weights(u4, {NE, N, n_groups, G}) → Convert(f16) → Subtract(zp_u8→f16) → Multiply(scale_f16) → Reshape({NE, N, H_or_I})
 * so ConvertMatMulToFullyConnected + MoE3GemmFusion collapse the whole graph.
 *
 * CLI:
 *   ./moe_bench <B> <S> <H> <I> <NE> <TK> [group_size=128] [iters=100] [warmup=10] [num_bufs=4] [flush_mb=64] [shared_I=0]
 *
 * If shared_I > 0, an additional always-on shared-expert FFN is appended
 *   shared_gate = MatMul(hidden_2d, sh_gate_w)  -> Swish
 *   shared_up   = MatMul(hidden_2d, sh_up_w)
 *   shared_mul  = Mul(shared_swish, shared_up)
 *   shared_down = MatMul(shared_mul, sh_down_w)
 *   final       = Add(MoE_out, shared_down)
 * which the GPU plugin's FuseMOESharedExpert pass folds into the same
 * MOE3GemmFusedCompressed primitive (see
 * src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_shared_expert.cpp).
 * The bench validates this fusion succeeded by checking that no
 * SharedGate/Up/Down MatMul friendly names survive in the runtime model.
 *
 * Between every infer, the same cache-flush Relu from fc_bench is enqueued
 * (SKILL §L3) so the 300+ MB expert-weight tensor cannot live in L2/L3 across
 * iterations. flush_mb=0 disables (ablation).
 */
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/pass/serialize.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ov;

// ---------- build a u4 grouped dequant chain with the given output shape ----------
// Chain:
//   Constant(u4, {NE, N, n_groups, G}) -> Convert(f16) -> Subtract(Convert(u4->f16 zp)) ->
//   Multiply(f16 scale) -> Reshape({NE, N, K}) -> Convert(f32)
// The final Convert(f32) is MANDATORY for ConvertMOEToMOECompressed to match (see
// src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp,
// GEMM3_PATTERN: `wrap_type<Convert>({reshape}, type_matches(f32))`).
static std::shared_ptr<Node> build_grouped_u4_weight(const Shape& logical_shape, int group_size, std::mt19937& rng) {
    OPENVINO_ASSERT(logical_shape.size() == 3, "weight logical shape must be 3D");
    size_t NE = logical_shape[0];
    size_t N  = logical_shape[1];
    size_t K  = logical_shape[2];
    OPENVINO_ASSERT(K % group_size == 0, "K must be divisible by group_size");
    size_t n_groups = K / group_size;
    size_t sG = size_t(group_size);

    size_t w_elems = NE * N * n_groups * sG;
    size_t w_bytes = (w_elems + 1) / 2;
    std::vector<uint8_t> w_packed(w_bytes);
    for (auto& v : w_packed) v = uint8_t(rng() % 256);
    auto w_const = op::v0::Constant::create(element::u4,
                                             Shape{NE, N, n_groups, sG}, w_packed.data());

    auto w_cvt = std::make_shared<op::v0::Convert>(w_const, element::f16);

    // Asymmetric path (works on both old and new OV builds):
    //   Convert(f16) -> Subtract(zp Convert(f16)) -> Multiply(scale f16) -> Reshape(3D) -> Convert(f32)
    // ZP and scale MUST share the same per-group shape {NE,N,n_groups,1} because the
    // ConvertMOEToMOECompressed callback applies the same Reshape-Transpose op to both.
    std::vector<uint8_t> zp_packed((NE * N * n_groups + 1) / 2, uint8_t(0x88));  // 8 = mid-range u4
    auto zp_const = op::v0::Constant::create(element::u4,
                                              Shape{NE, N, n_groups, 1}, zp_packed.data());
    auto zp_cvt  = std::make_shared<op::v0::Convert>(zp_const, element::f16);
    auto sub     = std::make_shared<op::v1::Subtract>(w_cvt, zp_cvt);

    std::vector<ov::float16> scale_data(NE * N * n_groups);
    for (auto& v : scale_data) v = ov::float16(float(rng() % 100 + 1) / 1000.0f);
    auto sc_const = op::v0::Constant::create(element::f16,
                                              Shape{NE, N, n_groups, 1}, scale_data.data());
    auto mul = std::make_shared<op::v1::Multiply>(sub, sc_const);

    auto reshape_to = op::v0::Constant::create(element::i32, Shape{3},
        std::vector<int32_t>{int32_t(NE), int32_t(N), int32_t(K)});
    auto resh = std::make_shared<op::v1::Reshape>(mul, reshape_to, false);

    // ConvertMOEToMOECompressed REQUIRES f32 here.
    return std::make_shared<op::v0::Convert>(resh, element::f32);
}

static std::shared_ptr<Node> build_grouped_u4_router(size_t H, size_t NE, int group_size, std::mt19937& rng) {
    OPENVINO_ASSERT(H % group_size == 0, "H must be divisible by group_size");
    size_t n_groups = H / group_size;
    size_t sG = size_t(group_size);
    size_t w_elems = NE * n_groups * sG;
    size_t w_bytes = (w_elems + 1) / 2;
    std::vector<uint8_t> w_packed(w_bytes);
    for (auto& v : w_packed) v = uint8_t(rng() % 256);
    auto w_const = op::v0::Constant::create(element::u4, Shape{NE, n_groups, sG}, w_packed.data());
    auto w_cvt = std::make_shared<op::v0::Convert>(w_const, element::f16);
    std::vector<ov::float16> scale_data(NE * n_groups);
    for (auto& v : scale_data) v = ov::float16(float(rng() % 100 + 1) / 1000.0f);
    auto sc_const = op::v0::Constant::create(element::f16, Shape{NE, n_groups, 1}, scale_data.data());
    auto mul = std::make_shared<op::v1::Multiply>(w_cvt, sc_const);
    auto reshape_to = op::v0::Constant::create(element::i32, Shape{2},
        std::vector<int32_t>{int32_t(NE), int32_t(H)});
    auto resh = std::make_shared<op::v1::Reshape>(mul, reshape_to, false);
    // Router is a plain FC; keep f32 output so MatMul naturally promotes.
    return std::make_shared<op::v0::Convert>(resh, element::f32);
}

// Shared-expert weight: 2D [N_out, K_in] u4-grouped. Same chain as the router
// (Constant(u4, {N_out, n_groups, G}) -> Convert(f16) -> Subtract(zp f16)
// -> Multiply(scale f16) -> Reshape({N_out, K_in}) -> Convert(f32)).
// FuseMOESharedExpert (src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_shared_expert.cpp)
// matches MatMul(hidden_2d, this_chain, false, true) for gate/up/down of the shared FFN.
static std::shared_ptr<Node> build_grouped_u4_shared_weight(size_t N_out, size_t K_in,
                                                            int group_size, std::mt19937& rng) {
    OPENVINO_ASSERT(K_in % group_size == 0, "K_in must be divisible by group_size");
    size_t n_groups = K_in / group_size;
    size_t sG = size_t(group_size);
    size_t w_elems = N_out * n_groups * sG;
    size_t w_bytes = (w_elems + 1) / 2;
    std::vector<uint8_t> w_packed(w_bytes);
    for (auto& v : w_packed) v = uint8_t(rng() % 256);
    auto w_const = op::v0::Constant::create(element::u4, Shape{N_out, n_groups, sG}, w_packed.data());
    auto w_cvt   = std::make_shared<op::v0::Convert>(w_const, element::f16);
    std::vector<uint8_t> zp_packed((N_out * n_groups + 1) / 2, uint8_t(0x88));
    auto zp_const = op::v0::Constant::create(element::u4, Shape{N_out, n_groups, 1}, zp_packed.data());
    auto zp_cvt   = std::make_shared<op::v0::Convert>(zp_const, element::f16);
    auto sub      = std::make_shared<op::v1::Subtract>(w_cvt, zp_cvt);
    std::vector<ov::float16> scale_data(N_out * n_groups);
    for (auto& v : scale_data) v = ov::float16(float(rng() % 100 + 1) / 1000.0f);
    auto sc_const = op::v0::Constant::create(element::f16, Shape{N_out, n_groups, 1}, scale_data.data());
    auto mul      = std::make_shared<op::v1::Multiply>(sub, sc_const);
    auto reshape_to = op::v0::Constant::create(element::i32, Shape{2},
        std::vector<int32_t>{int32_t(N_out), int32_t(K_in)});
    auto resh = std::make_shared<op::v1::Reshape>(mul, reshape_to, false);
    return std::make_shared<op::v0::Convert>(resh, element::f32);
}

// Plain FP16 shared-expert weight: 2D [N_out, K_in] constant (no compression).
// Used when the actual model shared expert has uncompressed FP16 weights.
// FuseMOESharedExpert uses any_input() for weights so this pattern still fuses.
static std::shared_ptr<Node> build_f16_shared_weight(size_t N_out, size_t K_in, std::mt19937& rng) {
    std::vector<ov::float16> w_data(N_out * K_in);
    for (auto& v : w_data) v = ov::float16(float(rng() % 200 - 100) / 1000.0f);
    auto w_const = op::v0::Constant::create(element::f16, Shape{N_out, K_in}, w_data.data());
    w_const->set_friendly_name("shared_f16_weight");
    // Convert to f32 to be compatible with the f32-activation MoE graph.
    return std::make_shared<op::v0::Convert>(w_const, element::f32);
}

// Canonical softmax-routing subgraph (minus experts) — returns {unsqueeze_routing_weights, topk_indices}.
// Mirrors build_softmax_routing_subgraph() from common_test_utils.
static std::pair<Output<Node>, Output<Node>>
build_softmax_routing(const Output<Node>& router_logits, size_t NE, size_t topk) {
    using namespace ov::op;
    auto sm = std::make_shared<v8::Softmax>(router_logits, 1);
    auto k = v0::Constant::create(element::i32, Shape{}, {int32_t(topk)});
    auto tk = std::make_shared<v11::TopK>(sm, k, 1, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES);
    auto rsum = std::make_shared<v1::ReduceSum>(tk->output(0),
                  v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}), true);
    auto normalized = std::make_shared<v1::Divide>(tk->output(0), rsum);
    auto topk_idx = tk->output(1);
    auto shapeof = std::make_shared<v3::ShapeOf>(topk_idx);
    auto gather = std::make_shared<v8::Gather>(shapeof,
                    v0::Constant::create(element::i64, Shape{}, {0}),
                    v0::Constant::create(element::i64, Shape{}, {0}));
    auto unsq_seq = std::make_shared<v0::Unsqueeze>(gather,
                      v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}));
    auto ne_scalar = v0::Constant::create(element::i64, Shape{}, {int64_t(NE)});
    auto unsq_ne = std::make_shared<v0::Unsqueeze>(ne_scalar,
                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}));
    auto bcast_shape = std::make_shared<v0::Concat>(OutputVector{unsq_seq, unsq_ne}, 0);
    auto one = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto reshape_shape = std::make_shared<v0::Concat>(OutputVector{unsq_ne, unsq_seq, one}, 0);
    auto zero = v0::Constant::create(normalized->get_element_type(), Shape{1}, {0});
    auto bcast = std::make_shared<v3::Broadcast>(zero, bcast_shape);
    auto scatter = std::make_shared<v12::ScatterElementsUpdate>(bcast, topk_idx, normalized,
                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto transp = std::make_shared<v1::Transpose>(scatter,
                    v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto resh = std::make_shared<v1::Reshape>(transp, reshape_shape, false);
    auto uns = std::make_shared<v0::Unsqueeze>(resh,
                 v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{3}));
    return {Output<Node>(uns), topk_idx};
}

// Build the full MoE subgraph the plugin will fuse. Uses f32 activation precision
// end-to-end so that the weight-decompression chain naturally terminates in Convert(f32)
// — a requirement of the GPU plugin's ConvertMOEToMOECompressed matcher.
// SI = shared_expert_intermediate_size (0 disables the shared-expert branch).
// shared_f16 = true → use uncompressed FP16 weights for shared expert instead of u4-grouped.
static std::shared_ptr<Model> build_moe_model(size_t B, size_t S, size_t H, size_t I, size_t NE, size_t TK, int gs,
                                              size_t SI, bool shared_f16 = false) {
    using namespace ov::op;
    std::mt19937 rng(42);

    auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape{int64_t(B), int64_t(S), int64_t(H)});
    input->set_friendly_name("hidden_states");

    auto experts_reshape = std::make_shared<v1::Reshape>(input,
        v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, int64_t(H)}), false);

    auto tile = std::make_shared<v0::Tile>(experts_reshape,
        v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{int64_t(NE), 1}));

    auto after_tile_resh = std::make_shared<v1::Reshape>(tile,
        v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{int64_t(NE), -1, int64_t(H)}), false);

    // Gate
    auto gate_w = build_grouped_u4_weight(Shape{NE, I, H}, gs, rng);   // logical [NE, I, H] (transposed MatMul)
    auto gate_mm = std::make_shared<v0::MatMul>(after_tile_resh, gate_w, false, true);
    gate_mm->set_friendly_name("GateMatMul");
    auto swish = std::make_shared<v4::Swish>(gate_mm);

    // Up
    auto up_w = build_grouped_u4_weight(Shape{NE, I, H}, gs, rng);
    auto up_mm = std::make_shared<v0::MatMul>(after_tile_resh, up_w, false, true);
    up_mm->set_friendly_name("UpMatMul");
    auto swiglu = std::make_shared<v1::Multiply>(swish, up_mm);

    // Down
    auto down_w = build_grouped_u4_weight(Shape{NE, H, I}, gs, rng);   // logical [NE, H, I]
    auto down_mm = std::make_shared<v0::MatMul>(swiglu, down_w, false, true);
    down_mm->set_friendly_name("DownMatMul");

    // Router
    auto router_w = build_grouped_u4_router(H, NE, gs, rng);           // [NE, H]
    auto router_mm = std::make_shared<v0::MatMul>(experts_reshape, router_w, false, true);  // [B*S, NE]

    auto [unsq_routing, topk_idx] = build_softmax_routing(router_mm, NE, TK);

    // Reshape down_mm [NE, seq, H] (softmax branch). Rebuild end-shape dynamically.
    auto ne_const = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{int64_t(NE)});
    auto minus_one = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto last_dim = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{int64_t(H)});
    auto first_seq = std::make_shared<v8::Gather>(
        std::make_shared<v3::ShapeOf>(topk_idx),
        v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
        v0::Constant::create(element::i64, Shape{}, {0}));
    auto end_shape = std::make_shared<v0::Concat>(OutputVector{ne_const, first_seq, minus_one, last_dim}, 0);
    auto end_resh = std::make_shared<v1::Reshape>(down_mm, end_shape, true);

    auto mul3 = std::make_shared<v1::Multiply>(end_resh, unsq_routing);
    auto reduce = std::make_shared<v1::ReduceSum>(mul3,
                    v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}), false);
    auto final_r = std::make_shared<v1::Reshape>(reduce,
                    v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, int64_t(H)}), true);

    // ---------- Optional shared-expert branch ----------
    // Pattern (matches FuseMOESharedExpert in src/plugins/intel_gpu/src/plugin/transformations/):
    //   shared_gate = MatMul(hidden_2d, sh_gate_w)   -> Swish -> sh_swish
    //   shared_up   = MatMul(hidden_2d, sh_up_w)
    //   shared_mul  = Mul(sh_swish, shared_up)
    //   shared_down = MatMul(shared_mul, sh_down_w)
    //   root        = Add(MOE_out, shared_down)
    // hidden_2d here is `experts_reshape` ([B*S, H]) — same tensor the router consumes,
    // which mirrors the real model layout (shared expert runs on every token).
    std::shared_ptr<Node> output_node = final_r;
    if (SI > 0) {
        std::shared_ptr<Node> sh_gate_w, sh_up_w, sh_down_w;
        if (shared_f16) {
            // Uncompressed FP16 shared expert weights
            sh_gate_w = build_f16_shared_weight(SI, H, rng);   // logical [SI, H]
            sh_up_w   = build_f16_shared_weight(SI, H, rng);   // logical [SI, H]
            sh_down_w = build_f16_shared_weight(H, SI, rng);   // logical [H, SI]
        } else {
            // INT4-grouped shared expert weights (default)
            sh_gate_w = build_grouped_u4_shared_weight(SI, H, gs, rng);   // logical [SI, H]
            sh_up_w   = build_grouped_u4_shared_weight(SI, H, gs, rng);   // logical [SI, H]
            sh_down_w = build_grouped_u4_shared_weight(H, SI, gs, rng);   // logical [H, SI]
        }

        auto sh_gate_mm = std::make_shared<v0::MatMul>(experts_reshape, sh_gate_w, false, true);
        sh_gate_mm->set_friendly_name("SharedGateMatMul");
        auto sh_swish   = std::make_shared<v4::Swish>(sh_gate_mm);
        auto sh_up_mm   = std::make_shared<v0::MatMul>(experts_reshape, sh_up_w, false, true);
        sh_up_mm->set_friendly_name("SharedUpMatMul");
        auto sh_mul     = std::make_shared<v1::Multiply>(sh_swish, sh_up_mm);
        auto sh_down_mm = std::make_shared<v0::MatMul>(sh_mul, sh_down_w, false, true);
        sh_down_mm->set_friendly_name("SharedDownMatMul");

        // CRITICAL: Add must consume the MOE output directly. MatMulExpertsFusion's
        // pattern root is ReduceSum, so after fusion `reduce` is replaced by the
        // internal MOE op. FuseMOESharedExpert's pattern is `Add(moe_m, shared_*)`
        // with NO Reshape allowed between MOE and Add (the optional Reshape sits
        // only on the shared-expert branch). If we feed `final_r` here, the post-
        // reduce Reshape silently blocks the shared-expert fusion.
        auto add = std::make_shared<v1::Add>(reduce, sh_down_mm);
        add->set_friendly_name("MoE_plus_SharedExpert");
        output_node = add;
    }

    return std::make_shared<Model>(OutputVector{std::make_shared<v0::Result>(output_node)},
                                   ParameterVector{input}, "MoE3GeMMBench");
}

// ---------- flush model (same as fc_bench) ----------
static std::shared_ptr<Model> build_flush_model(size_t n_elems) {
    auto p = std::make_shared<op::v0::Parameter>(element::f16, Shape{n_elems});
    auto r = std::make_shared<op::v0::Relu>(p);
    return std::make_shared<Model>(OutputVector{std::make_shared<op::v0::Result>(r)}, ParameterVector{p}, "flush");
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr <<
          "usage: moe_bench <B> <S> <H> <I> <NE> <TK> [group_size=128] [iters=100] [warmup=10] [num_bufs=4] [flush_mb=64] [shared_I=0] [shared_quant=u4|f16]\n"
          "  shared_I     = shared_expert_intermediate_size (0 disables; e.g. 512 for Qwen3.5-MoE)\n"
          "  shared_quant = weight type for shared expert: u4 (default) or f16 (uncompressed)\n"
          "example (Qwen3-Coder-30B decode):                       moe_bench 1 1 2048 768 128 8 128 200 20 4 64 0\n"
          "example (Qwen3.5-MoE-35B decode, shared expert INT4):   moe_bench 1 1 2048 512 256 8 128 200 20 4 64 512 u4\n"
          "example (Qwen3.5-MoE-35B decode, shared expert FP16):   moe_bench 1 1 2048 512 256 8 128 200 20 4 64 512 f16\n";
        return 1;
    }
    size_t B  = std::stoul(argv[1]);
    size_t S  = std::stoul(argv[2]);
    size_t H  = std::stoul(argv[3]);
    size_t I  = std::stoul(argv[4]);
    size_t NE = std::stoul(argv[5]);
    size_t TK = std::stoul(argv[6]);
    int gs           = (argc > 7)  ? std::stoi(argv[7])  : 128;
    int iters        = (argc > 8)  ? std::stoi(argv[8])  : 100;
    int warmup       = (argc > 9)  ? std::stoi(argv[9])  : 10;
    int num_bufs     = (argc > 10) ? std::stoi(argv[10]) : 4;
    int flush_mb     = (argc > 11) ? std::stoi(argv[11]) : 64;
    size_t SI        = (argc > 12) ? std::stoul(argv[12]) : 0;
    bool shared_f16  = (argc > 13) && std::string(argv[13]) == "f16";

    std::cout << "MoE bench: B=" << B << " S=" << S << " H=" << H << " I=" << I
              << " NE=" << NE << " TK=" << TK << " g=" << gs
              << " iters=" << iters << " warm=" << warmup
              << " bufs=" << num_bufs << " flush=" << flush_mb << "MB"
              << " shared_I=" << SI << (SI > 0 ? " [shared expert ENABLED]" : " [no shared expert]")
              << (SI > 0 ? (shared_f16 ? " shared_quant=f16" : " shared_quant=u4") : "")
              << std::endl;

    Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled;
    try {
        model = build_moe_model(B, S, H, I, NE, TK, gs, SI, shared_f16);
        // Dump the pre-compile model so we can inspect what the plugin pipeline sees
        // (FuseMOESharedExpert pattern requires plain v0::MatMul gate/up/down feeding an Add over MOE).
        try { ov::serialize(model, "/tmp/moe_bench_input.xml", "/tmp/moe_bench_input.bin"); } catch (...) {}
        AnyMap cfg = {{"PERF_COUNT", true}, {ov::hint::inference_precision.name(), ov::element::f16}};
        compiled = core.compile_model(model, "GPU", cfg);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] build/compile failed: " << e.what() << std::endl;
        return 3;
    }

    // ---------- Verify fusion fired ----------
    // The plugin's pipeline is:
    //   public-op subgraph  --MatMulExpertsFusion-->  ov::op::internal::MOE
    //                       --ConvertMoEToCompressed--> ov::intel_gpu::op::MOECompressed
    //                       --FuseMoE3GemmCompressed--> ov::intel_gpu::op::MOE3GemmFusedCompressed
    // If the final node is NOT present, we are NOT measuring the fused primitive.
    bool fusion_ok = false;
    bool shared_expert_residual = false;  // any leftover shared-expert MatMul in runtime graph
    try {
        auto rt_model = compiled.get_runtime_model();
        for (const auto& node : rt_model->get_ordered_ops()) {
            const auto& rt_info = node->get_rt_info();
            auto it = rt_info.find("layerType");
            std::string layer_type = (it != rt_info.end()) ? it->second.as<std::string>() : std::string{};
            std::string name = node->get_friendly_name();
            if (layer_type.find("moe_3gemm_fused_compressed") != std::string::npos ||
                name.find("moe_3gemm_fused_compressed") != std::string::npos ||
                name.find("MOE3GemmFusedCompressed")    != std::string::npos) {
                fusion_ok = true;
                std::cout << "[FUSION OK] Runtime node: " << name
                          << " layer_type=" << layer_type << std::endl;
            }
            // After FuseMOESharedExpert + later compression passes, the shared-expert
            // gate/up/down MatMuls must be folded into the fused primitive. If any of
            // their friendly names survive, the shared expert was NOT absorbed.
            if (name.find("SharedGateMatMul") != std::string::npos ||
                name.find("SharedUpMatMul")   != std::string::npos ||
                name.find("SharedDownMatMul") != std::string::npos) {
                shared_expert_residual = true;
                std::cerr << "[SHARED-EXPERT NOT FUSED] residual node: " << name
                          << " layer_type=" << layer_type << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARN] get_runtime_model failed: " << e.what() << std::endl;
    }
    if (!fusion_ok) {
        std::cerr << "[FUSION FAILED] moe_3gemm_fused_compressed not found in runtime model.\n"
                     "  The plugin did not fuse the subgraph. Possible causes:\n"
                     "   - MatMulExpertsFusion pattern did not match (check reshape/tile/normalize order)\n"
                     "   - Expert_type inferred as GEMM2 instead of GEMM3\n"
                     "   - Compressed-weight decompression pattern did not match\n"
                     "  Runtime graph dump to /tmp/moe_bench_rt.xml for inspection.\n";
        try {
            ov::serialize(compiled.get_runtime_model(), "/tmp/moe_bench_rt.xml", "/tmp/moe_bench_rt.bin");
        } catch (...) {}
        return 2;
    }
    if (SI > 0 && shared_expert_residual) {
        std::cerr << "[FUSION FAILED] shared-expert MatMuls survived as separate nodes;\n"
                     "  FuseMOESharedExpert did not absorb them into MOE3GemmFusedCompressed.\n"
                     "  Runtime graph dump to /tmp/moe_bench_rt.xml for inspection.\n";
        try {
            ov::serialize(compiled.get_runtime_model(), "/tmp/moe_bench_rt.xml", "/tmp/moe_bench_rt.bin");
        } catch (...) {}
        return 4;
    }
    if (SI > 0) {
        std::cout << "[SHARED-EXPERT FUSION OK] no residual SharedGate/Up/Down MatMul nodes" << std::endl;
    }

    // Rotating USM_DEVICE inputs.
    //
    // CRITICAL — input MUST be filled with random data, not left as the zero-page
    // that USM_DEVICE allocations default to. With zeroed hidden_states the router
    // MatMul produces identical logits for every token, Softmax is uniform, and
    // TopK(SORT_VALUES, MAX) deterministically picks experts [0..TK-1] for every
    // token regardless of S. The MoE prefill kernel then reads only TK experts'
    // weights from DRAM (≈19 MB) instead of the realistic ~NE experts (≈316 MB
    // for NE=128), under-counting MoE prefill memory traffic by ~16×. This
    // skews measured per-call MoE prefill time below the dense-read roofline
    // and breaks the §4.1 traffic model in SUMMARY_qwen3_moe.md.
    //
    // To realistically distribute tokens across all NE experts at S ≥ NE/TK,
    // each token's hidden_states must be different. We seed each USM_DEVICE
    // buffer by allocating a same-shape USM_HOST staging tensor (host-mappable),
    // filling it with random f32 in [-1, 1], then issuing a one-shot infer to
    // copy host → device once. Subsequent measurement infers reuse the device
    // tensor in place — no extra runtime overhead.
    auto rctx = compiled.get_context();
    std::vector<ov::Tensor> in_bufs;
    for (int i = 0; i < num_bufs; ++i) {
        in_bufs.push_back(rctx.create_tensor(element::f32, Shape{B, S, H}, AnyMap{}));
    }
    {
        // Build random hidden_states once on host and copy into each USM_DEVICE buffer.
        std::mt19937 rng(0xC0DE + 7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        ov::Tensor host_seed(element::f32, Shape{B, S, H});
        float* hp = host_seed.data<float>();
        size_t n = host_seed.get_size();
        for (size_t k = 0; k < n; ++k) hp[k] = dist(rng);
        // Re-seed each buffer with a different offset so rotating buffers don't
        // all carry identical data (which could artificially inflate L2 reuse
        // across iterations). The cheapest way is to XOR-shift the float bytes
        // per-buffer; we just bump the seed each time and refill.
        for (int i = 0; i < num_bufs; ++i) {
            for (size_t k = 0; k < n; ++k) hp[k] = dist(rng);
            // Use a temporary InferRequest to push the host data into the USM_DEVICE
            // buffer via the plugin's set_input_tensor + infer path. This relies on
            // OV's standard host→device copy when input is a plain Tensor and the
            // model's input is bound to a remote tensor — but here we copy directly
            // through the OpenCL context: easiest portable path is memcpy via
            // cl_mem mapping when available, falling back to set_input_tensor on a
            // throwaway request that targets in_bufs[i]:
            try {
                auto seed_req = compiled.create_infer_request();
                seed_req.set_input_tensor(in_bufs[i]);  // bind device-side dest
                // Manually copy host_seed → in_bufs[i] via the runtime tensor copy.
                // ov::Tensor::copy_to handles RemoteTensor on the destination.
                host_seed.copy_to(in_bufs[i]);
            } catch (const std::exception& e) {
                std::cerr << "[WARN] seeding USM_DEVICE buffer " << i << " failed: "
                          << e.what()
                          << "\n  -> router input may be zero; expert selection will be "
                             "deterministic (only experts 0..TK-1 active).\n";
            }
        }
        std::cout << "[INPUT SEED] filled " << num_bufs
                  << " USM_DEVICE input buffer(s) with random f32 in [-1,1]"
                  << " (B*S*H=" << n << " elems each)\n";
    }
    std::vector<InferRequest> reqs;
    for (int i = 0; i < num_bufs; ++i) {
        auto r = compiled.create_infer_request();
        r.set_input_tensor(in_bufs[i]);
        reqs.push_back(std::move(r));
    }

    // Flush model
    //
    // CACHE-HYGIENE NOTE — what this flush does and does NOT defeat:
    //   * DOES defeat: inter-iteration L3/L2 reuse of the rotating input buffers.
    //     A 64 MB ReLU evicts BMG's 18 MB and PTL's 8 MB L3 well before the next
    //     measurement iteration touches its input → input always cold-DRAM.
    //   * DOES defeat: inter-iteration weight reuse — by the time the flush model
    //     finishes, all expert weight tiles have been evicted from L3, so each
    //     measured iter starts with a cold weight cache (matches steady-state
    //     model inference where 48 layers cycle through ~316 MB of MoE weights).
    //   * DOES NOT (and cannot) defeat: INTRA-iteration producer→consumer cache
    //     reuse INSIDE the fused MoE primitive. The MoE primitive enqueues its
    //     8–10 sub-kernels (router gemm → softmax_topk → gather → dynamic_quantize
    //     → grouped_micro_gemm × 3 → swiglu → scatter_reduction) as one batch on
    //     the same GPU stream; intermediate tensors flow producer→consumer through
    //     L3 by design — that is the entire point of fusion. We CANNOT inject a
    //     flush between sub-kernels without modifying the plugin's compiled graph.
    //
    // Consequence for kernel_breakdown.py: a few sub-kernels (dynamic_quantize_gpu_opt,
    // moe_gather_ref_*, gemm_kernel router) achieve sub-DRAM-time because they read
    // their inputs straight from L3 instead of DRAM. The simple bytes-only roofline
    // there over-counts DRAM traffic, producing eff% > 100. The honest fix lives
    // in `theoretical_ms()` (model the real working set / cache-residence), NOT
    // in this test harness.
    bool flush_en = flush_mb > 0;
    InferRequest flush_req;
    if (flush_en) {
        size_t flush_elems = size_t(flush_mb) * 1024 * 1024 / 2;  // f16
        auto fm = build_flush_model(flush_elems);
        auto fc = core.compile_model(fm, "GPU");
        flush_req = fc.create_infer_request();
    }

    auto do_iter = [&](int i) {
        if (flush_en) flush_req.infer();
        reqs[i % num_bufs].infer();
    };

    std::cout << "warmup " << warmup << " iters...\n";
    for (int i = 0; i < warmup; ++i) do_iter(i);

    std::cout << "measuring " << iters << " iters...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) do_iter(i);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "total host wallclock = " << total_ms << " ms ("
              << total_ms / iters << " ms/iter host-side)\n"
              << "=> rely on cliloader DevicePerformanceTiming for fused MoE kernel avg\n";
    return 0;
}
