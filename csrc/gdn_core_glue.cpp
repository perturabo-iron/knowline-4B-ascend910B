/*
 * HOST-6 Optimization: C++ Glue Custom Operator for Qwen3.5 GDN Core on Ascend NPU.
 *
 * Replaces the per-layer Python + Triton dispatch loop inside
 * AscendGatedDeltaNetAttention._forward_core with a single C++ call that
 * orchestrates:
 *   1. fla_npu_stable::npu_causal_conv1d_fn (head-first BNSD output)
 *   2. Zero-copy Q/K/V slicing + fused QK L2-normalization (no Triton launch)
 *   3. GDN gating g & beta via native NPU ATen ops (no Triton launch)
 *   4. Zero-sync SSM state gather (skips completely via nullopt when no initial
 *      state exists; otherwise fuses select + transpose + copy per sequence)
 *   5. fla_npu_stable::npu_chunk_gated_delta_rule_fwd (pre-built host metadata)
 *   6. Zero-sync SSM state scatter (fuses transpose + copy directly into
 *      padded-stride ssm_state rows, no Triton launch, no 2MB temp buffer)
 *   7. In-place output copy into core_attn_out[:num_actual_tokens]
 *
 * Also provides gdn_decode_core for the pure-decode path.
 */

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace vllm_ascend_gdn {

namespace {

using CausalConv1dFnSchema = at::Tensor(
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t);

using CausalConv1dUpdateSchema = at::Tensor(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    int64_t,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    int64_t,
    int64_t,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    int64_t);

using ChunkGdnFwdRet = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>;

using ChunkGdnFwdSchema = ChunkGdnFwdRet(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    int64_t,
    double,
    int64_t,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    int64_t);

using RecurrentGdnSchema = at::Tensor(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    double,
    int64_t);

using NpuRmsNormSchema = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&,
    const at::Tensor&,
    double);

const c10::TypedOperatorHandle<CausalConv1dFnSchema>& get_causal_conv1d_fn_op() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("fla_npu_stable::npu_causal_conv1d_fn", "")
        .typed<CausalConv1dFnSchema>();
    return op;
}

const c10::TypedOperatorHandle<CausalConv1dUpdateSchema>& get_causal_conv1d_update_op() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("fla_npu_stable::npu_causal_conv1d_update", "")
        .typed<CausalConv1dUpdateSchema>();
    return op;
}

const c10::TypedOperatorHandle<ChunkGdnFwdSchema>& get_chunk_gdn_fwd_op() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("fla_npu_stable::npu_chunk_gated_delta_rule_fwd", "")
        .typed<ChunkGdnFwdSchema>();
    return op;
}

const c10::TypedOperatorHandle<RecurrentGdnSchema>& get_recurrent_gdn_op() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("fla_npu_stable::npu_recurrent_gated_delta_rule", "")
        .typed<RecurrentGdnSchema>();
    return op;
}

const c10::TypedOperatorHandle<NpuRmsNormSchema>& get_npu_rms_norm_op() {
    static const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("npu::npu_rms_norm", "")
        .typed<NpuRmsNormSchema>();
    return op;
}

inline int64_t get_npu_stream_ptr(bool use_nowait_stream) {
    // When libfla_npu_stable enqueues into torch_npu's task queue, reading the
    // stream with need_empty=false preserves exact FIFO queue ordering without
    // incurring the ~1 ms task-queue drain stall.
    aclrtStream raw_stream = c10_npu::getCurrentNPUStream().stream(!use_nowait_stream);
    return reinterpret_cast<int64_t>(raw_stream);
}

// Compute L2-normalized tensor along the last dimension D with eps=1e-6.
// Modes:
//   0 = aten_fp32     (0.0 bit-exact vs Triton l2norm_fwd)
//   1 = rms_norm_fp32 (single npu_rms_norm in fp32, <= 1 bf16 ULP)
//   2 = rms_norm_bf16 (single npu_rms_norm in bf16, fastest)
inline at::Tensor compute_l2norm(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& inv_sqrt_d_fp32,
    const c10::optional<at::Tensor>& inv_sqrt_d_bf16,
    int64_t l2norm_mode,
    double eps = 1e-6) {
    if (l2norm_mode == 2 && inv_sqrt_d_bf16.has_value() && inv_sqrt_d_bf16->defined()) {
        const double head_dim = static_cast<double>(x.size(-1));
        auto res = get_npu_rms_norm_op().call(x, *inv_sqrt_d_bf16, eps / head_dim);
        return std::get<0>(res);
    }
    if (l2norm_mode == 1 && inv_sqrt_d_fp32.has_value() && inv_sqrt_d_fp32->defined()) {
        const double head_dim = static_cast<double>(x.size(-1));
        at::Tensor xf = x.to(at::kFloat);
        auto res = get_npu_rms_norm_op().call(xf, *inv_sqrt_d_fp32, eps / head_dim);
        return std::get<0>(res).to(x.scalar_type());
    }
    // Mode 0: exact FP32 ATen formula matching Triton l2norm_fwd_kernel2
    at::Tensor xf = x.to(at::kFloat);
    at::Tensor sq_sum = at::mul(xf, xf).sum(-1, /*keepdim=*/true);
    sq_sum.add_(eps);
    sq_sum.rsqrt_();
    xf.mul_(sq_sum);
    return xf.to(x.scalar_type());
}

// Compute GDN gating tensors g ([1, T, Hv], fp32) and beta ([1, T, Hv], b.dtype).
// Note: aclnnSoftplus on Ascend 910B4 exhibits ~1.5e-5 abs / ~94% rel error for
// negative inputs. Using explicit log(1 + exp(clamp_max(x, 20))) with where()
// matches Triton fused_gdn_gating_kernel bit-for-bit (0.0 diff).
inline std::pair<at::Tensor, at::Tensor> compute_gdn_gating(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& neg_exp_A_log_fp32,
    const at::Tensor& dt_bias_fp32,
    bool bf16_sigmoid) {
    at::Tensor x = a.to(at::kFloat);
    x.add_(dt_bias_fp32);
    at::Tensor mask = x.le(20.0);
    at::Tensor sp = x.clamp_max(20.0);
    sp.exp_();
    sp.add_(1.0);
    sp.log_();
    at::Tensor g = at::where(mask, sp, x);
    g.mul_(neg_exp_A_log_fp32);
    at::Tensor beta = bf16_sigmoid
        ? at::sigmoid(b)
        : at::sigmoid(b.to(at::kFloat)).to(b.scalar_type());
    return {g.unsqueeze(0), beta.unsqueeze(0)};
}

}  // namespace

void gdn_prefill_core(
    const at::Tensor& mixed_qkv,
    const at::Tensor& b,
    const at::Tensor& a,
    at::Tensor& core_attn_out,
    const at::Tensor& conv_weight_T,
    const c10::optional<at::Tensor>& conv_bias,
    at::Tensor& conv_state,
    at::Tensor& ssm_state,
    const at::Tensor& neg_exp_A_log_fp32,
    const at::Tensor& dt_bias_fp32,
    const c10::optional<at::Tensor>& inv_sqrt_d_fp32,
    const c10::optional<at::Tensor>& inv_sqrt_d_bf16,
    const at::Tensor& query_start_loc,
    const at::Tensor& conv_cache_indices,
    const c10::optional<at::Tensor>& conv_has_initial_state,
    const at::Tensor& prefill_state_indices_cpu,
    const at::Tensor& prefill_has_initial_state_cpu,
    const at::Tensor& cu_seqlens_host,
    const at::Tensor& chunk_indices_host,
    const c10::optional<at::Tensor>& keep_mask_cpu,
    int64_t num_actual_tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t conv_head_num,
    int64_t activation_code,
    int64_t pad_slot_id,
    int64_t null_block_id,
    double scale,
    bool has_any_initial_state,
    bool output_final_state,
    bool use_nowait_stream,
    int64_t l2norm_mode,
    bool bf16_sigmoid,
    bool precomputed_gating,
    bool scatter_final_state) {
    RECORD_FUNCTION("vllm_ascend::gdn_prefill_core_cpp", std::vector<c10::IValue>{});

    if (num_actual_tokens <= 0) {
        return;
    }

    const int64_t stream_ptr = get_npu_stream_ptr(use_nowait_stream);

    // 1. Slice active tokens (avoiding copies when already full-length & contiguous).
    at::Tensor x_in = (mixed_qkv.size(0) == num_actual_tokens)
        ? mixed_qkv
        : mixed_qkv.narrow(0, 0, num_actual_tokens);
    if (!x_in.is_contiguous()) {
        x_in = x_in.contiguous();
    }
    at::Tensor b_in = (b.size(0) == num_actual_tokens) ? b : b.narrow(0, 0, num_actual_tokens);
    at::Tensor a_in = (a.size(0) == num_actual_tokens) ? a : a.narrow(0, 0, num_actual_tokens);

    // 2. Causal Conv1d (head-first output: [conv_head_num, T, D], naturally BNSD).
    const c10::optional<at::Tensor> no_tensor = c10::nullopt;
    at::Tensor conv_out = get_causal_conv1d_fn_op().call(
        x_in,
        conv_weight_T,
        conv_bias,
        conv_state,
        query_start_loc,
        conv_cache_indices,
        conv_has_initial_state,
        no_tensor,
        no_tensor,
        no_tensor,
        activation_code,
        pad_slot_id,
        null_block_id,
        conv_head_num,
        stream_ptr);

    // 3. Zero-copy Q+K slice and fused L2Norm; V stays untouched in BNSD!
    // conv_out has contiguous shape [2*num_k_heads + num_v_heads, T, D].
    const int64_t qk_heads = 2 * num_k_heads;
    at::Tensor qk = conv_out.narrow(0, 0, qk_heads);
    at::Tensor v = conv_out.narrow(0, qk_heads, num_v_heads).unsqueeze(0);

    at::Tensor qk_norm = compute_l2norm(qk, inv_sqrt_d_fp32, inv_sqrt_d_bf16, l2norm_mode);
    at::Tensor q = qk_norm.narrow(0, 0, num_k_heads).unsqueeze(0);
    at::Tensor k = qk_norm.narrow(0, num_k_heads, num_k_heads).unsqueeze(0);

    // 4. Gating (g: [1, T, Hv] fp32, beta: [1, T, Hv] v.dtype).
    at::Tensor g;
    at::Tensor beta;
    if (precomputed_gating) {
        g = a_in.unsqueeze(0);
        beta = b_in.unsqueeze(0);
    } else {
        auto gb = compute_gdn_gating(a_in, b_in, neg_exp_A_log_fp32, dt_bias_fp32, bf16_sigmoid);
        g = std::move(gb.first);
        beta = std::move(gb.second);
    }
    if (beta.scalar_type() != v.scalar_type()) {
        beta = beta.to(v.scalar_type());
    }

    // 5. Zero-sync SSM state gather directly into kernel layout [N_kern, Hv, K, V].
    //    When has_any_initial_state is false, pass nullopt (verified bit-exact on 910B4).
    const int64_t num_seqs = prefill_state_indices_cpu.numel();
    const int64_t* state_idx_ptr = prefill_state_indices_cpu.data_ptr<int64_t>();
    const bool* has_init_ptr = prefill_has_initial_state_cpu.data_ptr<bool>();
    const bool has_keep_mask = keep_mask_cpu.has_value() && keep_mask_cpu->defined();
    const bool* keep_ptr = has_keep_mask ? keep_mask_cpu->data_ptr<bool>() : nullptr;

    const int64_t num_kern_seqs = cu_seqlens_host.numel() - 1;
    c10::optional<at::Tensor> initial_state_kern = c10::nullopt;

    if (has_any_initial_state && num_kern_seqs > 0) {
        // ssm_state shape: [num_blocks, Hv, Dv, Dk] -> kernel wants [N_kern, Hv, Dk, Dv]
        const int64_t dv = ssm_state.size(2);
        const int64_t dk = ssm_state.size(3);
        at::Tensor init_buf = at::empty(
            {num_kern_seqs, num_v_heads, dk, dv},
            ssm_state.options());
        int64_t k_idx = 0;
        for (int64_t i = 0; i < num_seqs; ++i) {
            if (has_keep_mask && !keep_ptr[i]) {
                continue;
            }
            at::Tensor dst_row = init_buf.select(0, k_idx);
            if (has_init_ptr[i]) {
                // select(0, idx) is a zero-copy contiguous row view even when
                // ssm_state.stride(0) is padded; copy_ fuses transpose + copy!
                dst_row.copy_(ssm_state.select(0, state_idx_ptr[i]).transpose(-1, -2));
            } else {
                dst_row.zero_();
            }
            ++k_idx;
        }
        initial_state_kern = std::move(init_buf);
    }

    // 6. Core GDN chunked prefill kernel (layout=1 -> BNSD, state_v_first=false).
    auto fwd_ret = get_chunk_gdn_fwd_op().call(
        q,
        k,
        v,
        g,
        beta,
        initial_state_kern,
        cu_seqlens_host,
        chunk_indices_host,
        /*layout=*/1,
        scale,
        /*chunk_size=*/64,
        /*use_exp2=*/false,
        /*use_qk_l2norm_in_kernel=*/false,
        /*use_beta_sigmoid_in_kernel=*/false,
        /*allow_neg_eigval=*/false,
        /*disable_recompute=*/true,
        output_final_state,
        /*return_intermediate_states=*/false,
        /*state_v_first=*/false,
        stream_ptr);

    const at::Tensor& output = std::get<0>(fwd_ret);
    const c10::optional<at::Tensor>& final_state_opt = std::get<1>(fwd_ret);

    // 7. Zero-sync fused transpose + scatter directly into ssm_state rows.
    if (scatter_final_state && output_final_state && final_state_opt.has_value() && final_state_opt->defined()) {
        const at::Tensor& final_state = *final_state_opt;
        int64_t k_idx = 0;
        for (int64_t i = 0; i < num_seqs; ++i) {
            if (has_keep_mask && !keep_ptr[i]) {
                // Empty segments preserve their existing initial state in ssm_state.
                continue;
            }
            ssm_state.select(0, state_idx_ptr[i]).copy_(
                final_state.select(0, k_idx).transpose(-1, -2));
            ++k_idx;
        }
    }

    // 8. Write output into core_attn_out[:num_actual_tokens].
    at::Tensor out_squeezed = output.squeeze(0);
    if (core_attn_out.size(0) == num_actual_tokens) {
        core_attn_out.copy_(out_squeezed);
    } else {
        core_attn_out.narrow(0, 0, num_actual_tokens).copy_(out_squeezed);
    }
}

void gdn_decode_core(
    const at::Tensor& mixed_qkv,
    const at::Tensor& b,
    const at::Tensor& a,
    at::Tensor& core_attn_out,
    const at::Tensor& conv_weight_T,
    const c10::optional<at::Tensor>& conv_bias,
    at::Tensor& conv_state,
    at::Tensor& ssm_state,
    const at::Tensor& neg_exp_A_log_fp32,
    const at::Tensor& dt_bias_fp32,
    const c10::optional<at::Tensor>& inv_sqrt_d_fp32,
    const c10::optional<at::Tensor>& inv_sqrt_d_bf16,
    const at::Tensor& conv_cache_indices,
    const at::Tensor& query_start_loc,
    const at::Tensor& actual_seq_lengths,
    const at::Tensor& ssm_state_indices,
    int64_t num_actual_tokens,
    int64_t key_dim,
    int64_t value_dim,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t activation_code,
    int64_t null_block_id,
    double scale,
    bool use_nowait_stream,
    int64_t l2norm_mode,
    bool bf16_sigmoid,
    bool precomputed_gating) {
    RECORD_FUNCTION("vllm_ascend::gdn_decode_core_cpp", std::vector<c10::IValue>{});

    if (num_actual_tokens <= 0) {
        return;
    }

    const int64_t stream_ptr = get_npu_stream_ptr(use_nowait_stream);
    const c10::optional<at::Tensor> no_tensor = c10::nullopt;

    at::Tensor x_in = (mixed_qkv.size(0) == num_actual_tokens)
        ? mixed_qkv
        : mixed_qkv.narrow(0, 0, num_actual_tokens);
    at::Tensor b_in = (b.size(0) == num_actual_tokens) ? b : b.narrow(0, 0, num_actual_tokens);
    at::Tensor a_in = (a.size(0) == num_actual_tokens) ? a : a.narrow(0, 0, num_actual_tokens);

    at::Tensor conv_out = at::empty_like(x_in);
    conv_out = get_causal_conv1d_update_op().call(
        x_in,
        conv_state,
        conv_weight_T,
        conv_bias,
        activation_code,
        conv_cache_indices,
        no_tensor,
        query_start_loc,
        /*max_query_len=*/1,
        null_block_id,
        no_tensor,
        no_tensor,
        no_tensor,
        conv_out,
        stream_ptr);

    // Split token-major [T, 2*key_dim + value_dim] -> Q, K, V
    // Notice Q and K are adjacent: conv_out[:, :2*key_dim] can be L2-normed together!
    const int64_t num_k_heads = key_dim / head_k_dim;
    const int64_t num_v_heads = value_dim / head_v_dim;
    at::Tensor qk_flat = conv_out.narrow(1, 0, 2 * key_dim).contiguous().view({num_actual_tokens, 2 * num_k_heads, head_k_dim});
    at::Tensor qk_norm = compute_l2norm(qk_flat, inv_sqrt_d_fp32, inv_sqrt_d_bf16, l2norm_mode);
    at::Tensor q = qk_norm.narrow(1, 0, num_k_heads).contiguous();
    at::Tensor k = qk_norm.narrow(1, num_k_heads, num_k_heads).contiguous();
    at::Tensor v = conv_out.narrow(1, 2 * key_dim, value_dim).contiguous().view({num_actual_tokens, num_v_heads, head_v_dim});

    at::Tensor g;
    at::Tensor beta;
    if (precomputed_gating) {
        g = a_in;
        beta = b_in;
    } else {
        auto gb = compute_gdn_gating(a_in, b_in, neg_exp_A_log_fp32, dt_bias_fp32, bf16_sigmoid);
        g = gb.first.squeeze(0);
        beta = gb.second.squeeze(0);
    }

    at::Tensor out = get_recurrent_gdn_op().call(
        q,
        k,
        v,
        ssm_state,
        beta,
        actual_seq_lengths,
        ssm_state_indices,
        no_tensor,
        g,
        no_tensor,
        scale,
        stream_ptr);

    if (core_attn_out.size(0) == num_actual_tokens) {
        core_attn_out.copy_(out);
    } else {
        core_attn_out.narrow(0, 0, num_actual_tokens).copy_(out);
    }
}

}  // namespace vllm_ascend_gdn

TORCH_LIBRARY(_vllm_ascend_gdn, m) {
    m.def(
        "gdn_prefill_core("
        "Tensor mixed_qkv, Tensor b, Tensor a, Tensor(a!) core_attn_out, "
        "Tensor conv_weight_T, Tensor? conv_bias, Tensor(b!) conv_state, Tensor(c!) ssm_state, "
        "Tensor neg_exp_A_log_fp32, Tensor dt_bias_fp32, "
        "Tensor? inv_sqrt_d_fp32, Tensor? inv_sqrt_d_bf16, "
        "Tensor query_start_loc, Tensor conv_cache_indices, Tensor? conv_has_initial_state, "
        "Tensor prefill_state_indices_cpu, Tensor prefill_has_initial_state_cpu, "
        "Tensor cu_seqlens_host, Tensor chunk_indices_host, Tensor? keep_mask_cpu, "
        "int num_actual_tokens, int num_k_heads, int num_v_heads, int conv_head_num, "
        "int activation_code, int pad_slot_id, int null_block_id, float scale, "
        "bool has_any_initial_state, bool output_final_state, bool use_nowait_stream, "
        "int l2norm_mode, bool bf16_sigmoid, bool precomputed_gating, bool scatter_final_state) -> ()");
    m.def(
        "gdn_decode_core("
        "Tensor mixed_qkv, Tensor b, Tensor a, Tensor(a!) core_attn_out, "
        "Tensor conv_weight_T, Tensor? conv_bias, Tensor(b!) conv_state, Tensor(c!) ssm_state, "
        "Tensor neg_exp_A_log_fp32, Tensor dt_bias_fp32, "
        "Tensor? inv_sqrt_d_fp32, Tensor? inv_sqrt_d_bf16, "
        "Tensor conv_cache_indices, Tensor query_start_loc, "
        "Tensor actual_seq_lengths, Tensor ssm_state_indices, "
        "int num_actual_tokens, int key_dim, int value_dim, int head_k_dim, int head_v_dim, "
        "int activation_code, int null_block_id, float scale, "
        "bool use_nowait_stream, int l2norm_mode, bool bf16_sigmoid, bool precomputed_gating) -> ()");
}

TORCH_LIBRARY_IMPL(_vllm_ascend_gdn, PrivateUse1, m) {
    m.impl("gdn_prefill_core", &vllm_ascend_gdn::gdn_prefill_core);
    m.impl("gdn_decode_core", &vllm_ascend_gdn::gdn_decode_core);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ glue custom operators for Qwen3.5 GDN Core on Ascend NPU (HOST-6)";
    m.def("gdn_prefill_core", &vllm_ascend_gdn::gdn_prefill_core,
          "Fused C++ glue for GDN prefill core",
          py::arg("mixed_qkv"), py::arg("b"), py::arg("a"), py::arg("core_attn_out"),
          py::arg("conv_weight_T"), py::arg("conv_bias"), py::arg("conv_state"), py::arg("ssm_state"),
          py::arg("neg_exp_A_log_fp32"), py::arg("dt_bias_fp32"),
          py::arg("inv_sqrt_d_fp32"), py::arg("inv_sqrt_d_bf16"),
          py::arg("query_start_loc"), py::arg("conv_cache_indices"), py::arg("conv_has_initial_state"),
          py::arg("prefill_state_indices_cpu"), py::arg("prefill_has_initial_state_cpu"),
          py::arg("cu_seqlens_host"), py::arg("chunk_indices_host"), py::arg("keep_mask_cpu"),
          py::arg("num_actual_tokens"), py::arg("num_k_heads"), py::arg("num_v_heads"), py::arg("conv_head_num"),
          py::arg("activation_code"), py::arg("pad_slot_id"), py::arg("null_block_id"), py::arg("scale"),
          py::arg("has_any_initial_state"), py::arg("output_final_state"), py::arg("use_nowait_stream"),
          py::arg("l2norm_mode"), py::arg("bf16_sigmoid"), py::arg("precomputed_gating") = false,
          py::arg("scatter_final_state") = true);
    m.def("gdn_decode_core", &vllm_ascend_gdn::gdn_decode_core,
          "Fused C++ glue for GDN decode core",
          py::arg("mixed_qkv"), py::arg("b"), py::arg("a"), py::arg("core_attn_out"),
          py::arg("conv_weight_T"), py::arg("conv_bias"), py::arg("conv_state"), py::arg("ssm_state"),
          py::arg("neg_exp_A_log_fp32"), py::arg("dt_bias_fp32"),
          py::arg("inv_sqrt_d_fp32"), py::arg("inv_sqrt_d_bf16"),
          py::arg("conv_cache_indices"), py::arg("query_start_loc"),
          py::arg("actual_seq_lengths"), py::arg("ssm_state_indices"),
          py::arg("num_actual_tokens"), py::arg("key_dim"), py::arg("value_dim"),
          py::arg("head_k_dim"), py::arg("head_v_dim"),
          py::arg("activation_code"), py::arg("null_block_id"), py::arg("scale"),
          py::arg("use_nowait_stream"), py::arg("l2norm_mode"), py::arg("bf16_sigmoid"),
          py::arg("precomputed_gating") = false);
}

