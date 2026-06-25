# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Replicates OpenVINO intel_gpu's OCL JIT codegen for the extracted GGUF kernels so the
*verbatim* .cl files compile and run identically to how OV builds them at runtime.

Two things OV does, reproduced here 1:1:

1. The build preamble (graph/impls/ocl_v2/utils/kernel_generator.cpp build_code()):
     #define FUNC(name)            _##name##_<entry>
     #define FUNC_CALL(name)       _##name##_<entry>
     #define CONST_ARRAY_DECL(name) __constant size_t  _##name##_<entry> []
     #define CONST_ARRAY_REF(name)  _##name##_<entry>
     #define KERNEL(name)          __kernel void <entry>
     #define KERNEL_ID             <entry>
     ... then one `#define <name> <value>` per JitConstant ...
   followed by the kernel template text (which #includes common.cl etc).

2. The per-kernel JitConstants emitted by each *Generator::get_jit_constants in
   gguf/fc_gguf_opt.cpp:
     - FCGGUFOptGenerator        (fc_gguf_opt.cl)
     - FCGGUFTranscodeGenerator  (fc_gguf_transcode.cl)
     - FCGGUFDp4aGenerator       (fc_gguf_dp4a.cl)
     - FCGGUFPrequantGenerator   (fc_gguf_prequant.cl)
   including make_type_jit_constants() (INPUT0_TYPE / OUTPUT_TYPE / TO_*_TYPE ...) and the
   layout GET_INDEX macros.

Layout note (faithfulness): the GGUF FC activation [BM,K] and output [BM,N] are plain
contiguous row-major buffers. OV reaches them through INPUT0_GET_INDEX(b,f,y,x) after
decomposing bm into (b,f) via INPUT0_FEATURE_NUM; for a contiguous bfyx tensor with
y-size=K, x-size=1 that index reduces to exactly bm*K + k. The dp4a and prequant kernels
hardcode `bm*K_SIZE + k` / `bm*N_SIZE + n`, confirming OV's intended row-major contiguity.
We therefore emit the reduced GET_INDEX macros directly (FEATURE_NUM=1, b=bm, f=0), which
produce byte-identical addressing to OV for this primitive.
"""

# ----------------------------------------------------------------------------------------
# OCL scalar type table -- mirrors jitter.cpp make_type_jit_constants / to_ocl_type.
# ----------------------------------------------------------------------------------------
# et -> (ocl_type, convert_fn, as_fn, val_max, val_min, type_size, is_fp)
_TYPE_TABLE = {
    "f16":  ("half",  "convert_half",  "as_half",  "HALF_MAX", "-HALF_MAX", 2, True),
    "f32":  ("float", "convert_float", "as_float", "FLT_MAX",  "-FLT_MAX",  4, True),
    "i8":   ("char",  "convert_char",  "as_char",  "CHAR_MAX", "CHAR_MIN",  1, False),
    "u8":   ("uchar", "convert_uchar", "as_uchar", "UCHAR_MAX","0",         1, False),
    "i16":  ("short", "convert_short", "as_short", "SHRT_MAX", "SHRT_MIN",  2, False),
    "u16":  ("ushort","convert_ushort","as_ushort","USHRT_MAX","0",         2, False),
    "i32":  ("int",   "convert_int",   "as_int",   "INT_MAX",  "INT_MIN",   4, False),
    "u32":  ("uint",  "convert_uint",  "as_uint",  "UINT_MAX", "0",         4, False),
}


def make_type_jit_constants(name, et):
    """Mirror make_type_jit_constants(name, element::Type): name in {INPUT0,OUTPUT,...}."""
    ocl, conv, asf, vmax, vmin, tsz, is_fp = _TYPE_TABLE[et]
    return {
        "%s_TYPE" % name: ocl,
        "%s_VAL_MAX" % name: vmax,
        "%s_VAL_MIN" % name: vmin,
        "%s_VAL_ONE" % name: "(%s) 1" % ocl,
        "%s_VAL_ZERO" % name: "(%s) 0" % ocl,
        "TO_%s_TYPE(v)" % name: "%s(v)" % conv,
        "TO_%s_TYPE_SAT(v)" % name: "%s_sat(v)" % conv,
        "AS_%s_TYPE(v)" % name: "%s(v)" % asf,
        "%s_MAX_FUNC" % name: "fmax" if is_fp else "max",
        "%s_MIN_FUNC" % name: "fmin" if is_fp else "min",
        "%s_ABS_FUNC" % name: "fabs" if is_fp else "abs",
        "%s_TYPE_SIZE" % name: str(tsz),
        "%s_IS_FP" % name: "1" if is_fp else "0",
    }


def _layout_index_macros(name, axis_size_macro):
    """Reduced contiguous-bfyx GET_INDEX for a [BM, <axis_size>] row-major tensor.

    GET_INDEX(b,f,y,x) = ((b*FEATURE_NUM + f) * AXIS + y).  With FEATURE_NUM=1 and the
    caller passing b=bm, f=0, y=k/n, x=0 this is bm*AXIS + k -- identical to OV for a
    contiguous bfyx activation/output of this FC primitive.
    """
    return {
        "%s_FEATURE_NUM" % name: "1",
        "%s_GET_INDEX(b,f,y,x)" % name: "(((b)*%s_FEATURE_NUM + (f))*%s + (y))" % (name, axis_size_macro),
    }


# ----------------------------------------------------------------------------------------
# Build preamble -- mirrors CodeBuilder decoration_macro + value_macro emission order.
# ----------------------------------------------------------------------------------------
def _decoration_macro(name, prefix, postfix, name_prefix=""):
    # Exactly reproduces CodeBuilder::decoration_macro formatting:
    #   #define <name>(name) <prefix> <name_prefix>_##name[##_<postfix>]
    sep = "##_" if postfix else ""
    return "#define %s(name) %s %s_##name%s%s\n" % (name, prefix, name_prefix, sep, postfix)


def build_preamble(entry_point, consts):
    """Return the full JIT macro preamble string for one kernel build.

    `consts` is an ordered dict of JitConstant name->value (name may include a (args) part
    for function-like macros, e.g. 'TO_OUTPUT_TYPE(v)').
    """
    s = []
    s.append("//==================================================== JIT preamble (OV-faithful)")
    s.append("// Kernel entry point: %s" % entry_point)
    s.append(_decoration_macro("FUNC", "", entry_point).rstrip("\n"))
    s.append(_decoration_macro("FUNC_CALL", "", entry_point).rstrip("\n"))
    s.append(_decoration_macro("CONST_ARRAY_DECL", "__constant size_t ", entry_point + " []").rstrip("\n"))
    s.append(_decoration_macro("CONST_ARRAY_REF", "", entry_point).rstrip("\n"))
    s.append("#define KERNEL(name) __kernel void %s" % entry_point)
    s.append("#define KERNEL_ID %s" % entry_point)
    for k, v in consts.items():
        s.append("#define %s %s" % (k, v))
    s.append("//==================================================== end JIT preamble")
    return "\n".join(s) + "\n"


# ----------------------------------------------------------------------------------------
# Per-kernel JitConstant builders -- each mirrors its *Generator::get_jit_constants in
# gguf/fc_gguf_opt.cpp. Returns (entry_point, ordered-dict-of-consts).
# ----------------------------------------------------------------------------------------
from collections import OrderedDict
from gguf_ref import block_elem, block_bytes, GGUF_TYPE_FLAG, GGUF_TRANSCODE_TARGET, GGUF_REQUANT_GROUP

GGUF_GEMV_SG_SIZE = 16  # mirrors GGUF_GEMV_SG_SIZE in fc_gguf_opt.cpp


def opt_jit(name, K, N, M, in0_dt="f16", out_dt="f16", dynamic=False):
    """FCGGUFOptGenerator: the memory-bound GEMV (handles any M)."""
    ep = "fc_gguf_opt_%s_K%d_N%d" % (name, K, N)
    ep += "__sa" if dynamic else ("_M%d" % M)
    c = OrderedDict()
    if dynamic:
        c["OPTIONAL_SHAPE_INFO_ARG"] = "__global const int* shape_info,"
        c["OPTIONAL_SHAPE_INFO_TENSOR"] = "shape_info,"
        c["IS_DYNAMIC"] = "1"
    else:
        c["OPTIONAL_SHAPE_INFO_ARG"] = ""
        c["OPTIONAL_SHAPE_INFO_TENSOR"] = ""
    # K_SIZE/N_SIZE first so the GET_INDEX macros below can reference them.
    c["K_SIZE"] = str(K)
    c["N_SIZE"] = str(N)
    c.update(make_type_jit_constants("INPUT0", in0_dt))
    c.update(make_type_jit_constants("OUTPUT", out_dt))
    c.update(_layout_index_macros("INPUT0", "K_SIZE"))
    c.update(_layout_index_macros("OUTPUT", "N_SIZE"))
    c["GGUF_BLOCK_ELEM"] = str(block_elem(name))
    c["GGUF_BLOCK_BYTES"] = str(block_bytes(name))
    c["SG_SIZE"] = str(GGUF_GEMV_SG_SIZE)
    c[GGUF_TYPE_FLAG[name]] = "1"
    return ep, c


def transcode_jit(name, K, N):
    """FCGGUFTranscodeGenerator: GGUF block -> i4/i8 + f16 scale (prefill path)."""
    ep = "fc_gguf_transcode_%s_K%d_N%d" % (name, K, N)
    to_i4, qmax = GGUF_TRANSCODE_TARGET[name]
    c = OrderedDict()
    c["OPTIONAL_SHAPE_INFO_ARG"] = ""
    c["OPTIONAL_SHAPE_INFO_TENSOR"] = ""
    c["K_SIZE"] = str(K)
    c["N_SIZE"] = str(N)
    c["GGUF_BLOCK_ELEM"] = str(block_elem(name))
    c["GGUF_BLOCK_BYTES"] = str(block_bytes(name))
    c["REQUANT_GROUP"] = str(GGUF_REQUANT_GROUP)
    c["TRANSCODE_TO_I4"] = "1" if to_i4 else "0"
    c["QMAX"] = str(qmax)
    c[GGUF_TYPE_FLAG[name]] = "1"
    return ep, c


def dp4a_jit(name, K, N, out_dt="f16", nrow=None):
    """FCGGUFDp4aGenerator: Q5_K/Q6_K SWAR dp4a GEMV (int8 activation decode path)."""
    assert name in ("gguf_q5_k", "gguf_q6_k"), "dp4a path only exists for Q5_K / Q6_K"
    ep = "fc_gguf_dp4a_%s_K%d_N%d" % (name, K, N)
    if nrow is None:
        nrow = 4 if name == "gguf_q6_k" else 1  # mirrors gguf_q6k_nrow() default
    c = OrderedDict()
    c["OPTIONAL_SHAPE_INFO_ARG"] = ""
    c["OPTIONAL_SHAPE_INFO_TENSOR"] = ""
    c.update(make_type_jit_constants("OUTPUT", out_dt))
    c["K_SIZE"] = str(K)
    c["N_SIZE"] = str(N)
    c["GGUF_BLOCK_ELEM"] = str(block_elem(name))
    c["GGUF_BLOCK_BYTES"] = str(block_bytes(name))
    c["SG_SIZE"] = str(GGUF_GEMV_SG_SIZE)
    c["NROW"] = str(nrow)
    c[GGUF_TYPE_FLAG[name]] = "1"
    return ep, c, nrow


def prequant_jit(name, K, N, in0_dt="f16"):
    """FCGGUFPrequantGenerator: activation -> int8 + per-32 f32 scale."""
    ep = "fc_gguf_prequant_%s_K%d_N%d" % (name, K, N)
    c = OrderedDict()
    c["OPTIONAL_SHAPE_INFO_ARG"] = ""
    c["OPTIONAL_SHAPE_INFO_TENSOR"] = ""
    c.update(make_type_jit_constants("INPUT0", in0_dt))
    c["K_SIZE"] = str(K)
    return ep, c


# OV build options for Intel devices -- mirrors KernelGenerator::get_build_options exactly:
#   -cl-mad-enable                (always, for Intel vendor)
#   -cl-std=CL3.0                 (devices with work-group collective functions; the B580 has
#                                  them. CL3.0 is also what exposes the cl_khr_integer_dot_product
#                                  dot_acc_sat_4x8packed_* builtins used by the dp4a kernel.)
# Returned as a list for pyopencl Program.build(options=[...]).
OV_BUILD_OPTIONS = ["-cl-mad-enable", "-cl-std=CL3.0"]


if __name__ == "__main__":
    ep, c = opt_jit("gguf_q4_k", 1024, 512, 1)
    print(build_preamble(ep, c)[:1200])
