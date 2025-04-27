import logging
from typing import Callable, Literal, Optional, get_args

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

logger = logging.getLogger(__name__)



MambaBlock_Hook_Points = Literal[
    "ssm_after_up_proj",
    "ssm_after_conv1D",
    "ssm_after_silu",
    "ssm_after_ssm",
    "mlp_after_up_proj",
    "mlp_after_silu",
    "before_down_proj",
    "after_down_proj",  # the output of the mamba block #! Not the residual
]


def MambaBlockForwardPatcher(
    patch_spec: Optional[dict[int, torch.Tensor]] = None,
    patch_hook: Optional[
        MambaBlock_Hook_Points
    ] = None,  # If None => do not patch, return the original output
    retainer: Optional[
        dict
    ] = None,  # if a dictionary is passed, will retain all the activations[patch_idx] at different hook points
) -> Callable:
    # TODO: Assumes a single prompt for now. Should we consider batching?
    """
    Returns a replacement for the `forward()` method of `MambaBlock` to patch activations at different steps.
    """
    if patch_hook is None:
        assert (
            patch_spec is None
        ), "Need to specify `patch_hook` if `patch_spec` is not None"
    else:
        assert patch_hook in get_args(
            MambaBlock_Hook_Points
        ), f"Unknown `{patch_hook=}`, should be one of {get_args(MambaBlock_Hook_Points)}"
        assert isinstance(
            patch_spec, dict
        ), f"Need to specify `patch_spec` as a dictionary for `{patch_hook=}`"
    if retainer is not None:
        assert isinstance(retainer, dict)

    def forward_patcher(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )
        x = x.clone()
        # ------------------------------------------------------
        if patch_hook == "ssm_after_up_proj":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        elif patch_hook == "mlp_after_up_proj":
            for patch_idx, patch_vector in patch_spec.items():
                res[:, patch_idx] = patch_vector.to(res.dtype).to(res.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_up_proj"] = x.detach().clone()
            retainer["mlp_after_up_proj"] = res.detach().clone()
        # ------------------------------------------------------

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")
        # ------------------------------------------------------
        if patch_hook == "ssm_after_conv1D":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_conv1D"] = x.detach().clone()
        # ------------------------------------------------------

        x = F.silu(x)
        # ------------------------------------------------------
        if patch_hook == "ssm_after_silu":
            for patch_idx, patch_vector in patch_spec.items():
                x[:, patch_idx] = patch_vector.to(x.dtype).to(x.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_silu"] = x.detach().clone()
        # ------------------------------------------------------

        y = self.ssm(x)
        # ------------------------------------------------------
        if patch_hook == "ssm_after_ssm":
            for patch_idx, patch_vector in patch_spec.items():
                y[:, patch_idx] = patch_vector.to(y.dtype).to(y.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["ssm_after_ssm"] = y.detach().clone()
        # ------------------------------------------------------

        res = F.silu(res)
        # ------------------------------------------------------
        if patch_hook == "mlp_after_silu":
            for patch_idx, patch_vector in patch_spec.items():
                res[:, patch_idx] = patch_vector.to(res.dtype).to(res.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["mlp_after_silu"] = res.detach().clone()
        # ------------------------------------------------------

        y = y * res
        # ------------------------------------------------------
        if patch_hook == "before_down_proj":
            for patch_idx, patch_vector in patch_spec.items():
                y[:, patch_idx] = patch_vector.to(y.dtype).to(y.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["before_down_proj"] = y.detach().clone()
        # ------------------------------------------------------

        output = self.out_proj(y)
        # ------------------------------------------------------
        if patch_hook == "after_down_proj":
            for patch_idx, patch_vector in patch_spec.items():
                output[:, patch_idx] = patch_vector.to(output.dtype).to(output.device)
        # ------------------------------------------------------
        if retainer is not None:
            retainer["after_down_proj"] = output.detach().clone()
        # ------------------------------------------------------

        return output

    return forward_patcher

Mamba2Block_Hook_Points = Literal[
    "ssm_after_ssm",
    "after_down_proj",
]

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

def Mamba2BlockForwardPatcher(
    patch_spec: Optional[dict[int, torch.Tensor]] = None,
    patch_hook: Optional[Mamba2Block_Hook_Points] = None,
    retainer: Optional[dict] = None,
) -> Callable:
    
    def forward_patcher(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        # For now, skip CUDA fast path and assume non-cached, float32 mode
    
        # --- Step 1: Input Projection ---
        # projected_states = self.in_proj(hidden_states)
        
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
    
        # --- Step 2: Shape & Dim Setup ---
        batch_size, seq_len, _ = hidden_states.shape
        
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
    
    
        A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        # --- Step 3: Split Streams ---
        # z0, x0, gate, hidden_states_BC, dt = projected_states.split(
        #     [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        # )
        
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
                    [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )

                # 2. Convolution sequence transformation
                # Init cache
        if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(
                        hidden_states_B_C_transposed,
                        (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                    )
    
        # --- Step 4: Conv Block ---
        # hidden_states_BC = self.act(self.conv1d(hidden_states_BC.transpose(1, 2)).transpose(1, 2)[..., :L])
        
        if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                    )
        else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
    

        hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
        )
        
    
        # --- Step 5: SSM Block ---
        scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

        if patch_hook == "ssm_after_ssm":
            for idx, vec in patch_spec.items():
                scan_output[:, idx] = vec.to(scan_output.device)
        if retainer is not None:
            retainer["ssm_after_ssm"] = scan_output.detach().clone()
            
        
        if patch_hook == "mlp_after_silu":
            for idx, vec in patch_spec.items():
                gate[:, idx] = vec.to(gate.device)
        if retainer is not None:
            retainer["mlp_after_silu"] = gate.detach().clone()

    
        # --- Step 6: MLP Block ---
         # Init cache
        if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

        scan_output = scan_output.view(batch_size, seq_len, -1)
        # Multiply "gate" branch and apply extra normalization layer
        scan_output = self.norm(scan_output, gate)

        # 4. Final linear projection
        out = self.out_proj(scan_output)
        
        if patch_hook == "after_down_proj":
            for idx, vec in patch_spec.items():
                out[:, idx] = vec.to(out.device)
        if retainer is not None:
            retainer["after_down_proj"] = out.detach().clone()
            
        
        return out

  
    # def forward_patcher(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, **kwargs):
        
    #     seqlen_og = seqlen
    #     if seqlen is None:
    #         B, L, _ = u.shape
    #     else:
    #         B = u.shape[0] // seqlen
    #         L = seqlen
    
    #     print("[DEBUG] In patcher: self class =", self.__class__)
    #     assert False
        
    #     # (B, L, 18560)
    #     if hasattr(self, "in_proj"):
    #         zxbcdt = self.in_proj(u)
    #     elif hasattr(self, "mixer") and hasattr(self.mixer, "in_proj"):
    #         zxbcdt = self.mixer.in_proj(u)
    #     else:
    #         raise AttributeError("No in_proj found in current context.")


    #     if seqlen_og is not None:
    #         zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
    
    #     # Extract from attributes
    #     d_ssm = self.num_heads * self.head_dim         # 128 * 64 = 8192
    #     d_state = self.ssm_state_size                  # 128
    #     d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * self.n_groups * d_state - self.num_heads) // 2
    #     d_conv = self.conv1d.weight.shape[-1]          # Kernel width, usually 4
    #     g = self.n_groups
    #     n_heads = self.num_heads
    #     head_dim = self.head_dim
    
    #     # Split into functional streams
    #     z0, x0, z, xBC, dt = torch.split(
    #         zxbcdt,
    #         [d_mlp, d_mlp, d_ssm, d_ssm + 2 * g * d_state, n_heads],
    #         dim=-1
    #     )

    
    #     # Convolution branch
    #     xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(d_conv - 1)])
    #     x, B_, C = torch.split(xBC, [d_ssm, g * d_state, g * d_state], dim=-1)
        
    #     # Check if D has head dimension
    #     if self.D.ndim == 2 and self.D.shape[0] == self.num_heads * self.head_dim:
    #         D_reshaped = rearrange(self.D, "(h p) -> h p", p=self.head_dim)
    #     else:
    #         D_reshaped = self.D
            
    #     z_for_scan = ( None if self.rms_norm else rearrange(z, "b l (h p) -> b l h p", p=head_dim))
    #     dt_limit = (self.time_step_min, self.time_step_max)

    #     # SSM branch
    #     y_ssm = mamba_chunk_scan_combined(
    #         rearrange(x, "b l (h p) -> b l h p", p=head_dim),
    #         dt,
    #         -torch.exp(self.A_log.float()),
    #         rearrange(B_, "b l (g n) -> b l g n", g=g),
    #         rearrange(C, "b l (g n) -> b l g n", g=g),
    #         chunk_size=self.chunk_size,
    #         D=D_reshaped,
    #         z=z_for_scan,
    #         dt_bias=self.dt_bias,
    #         dt_softplus=True,
    #         seq_idx=seq_idx,
    #         cu_seqlens=cu_seqlens,
    #         dt_limit=dt_limit,
    #     )
    #     y_ssm = rearrange(y_ssm, "b l h p -> b l (h p)")
    
    #     if patch_hook == "ssm_after_ssm":
    #         for idx, vec in patch_spec.items():
    #             y_ssm[:, idx] = vec.to(y_ssm.device)
    #     if retainer is not None:
    #         retainer["ssm_after_ssm"] = y_ssm.detach().clone()
    
    #     # MLP branch
    #     y_mlp = F.silu(z0) * x0
    #     if patch_hook == "mlp_after_silu":
    #         for idx, vec in patch_spec.items():
    #             y_mlp[:, idx] = vec.to(y_mlp.device)
    #     if retainer is not None:
    #         retainer["mlp_after_silu"] = y_mlp.detach().clone()
    
    #     # Concatenate and project
    #     y = torch.cat([y_mlp, y_ssm], dim=-1)
    #     if seqlen_og is not None:
    #         y = rearrange(y, "b l d -> (b l) d")
    
    #     out = self.out_proj(y)
        
    #     if patch_hook == "after_down_proj" and patch_spec is not None:
    #         for idx, vec in patch_spec.items():
        
    #             # --- Sanity Check: Inject corruption first ---
    #             # print(f"[Corrupt] Overwriting out[:, {idx}] with noise before patching")
        
    #             # Save corrupted version to compute delta
    #             # before = out[:, idx].clone()
                
    #             # print(f"[Pre-patch] Δ(clean vs restore) = {(out[0, idx] - vec.to(out.device)).abs().mean().item():.6f}")
    #             # print(f"[Pre-patch] Δ(corrupt vs restored) = {(out[1, idx] - vec.to(out.device)).abs().mean().item():.6f}")


        
    #             # Apply patch
    #             out[:, idx] = vec.to(out.device)
    #             # print(f"[Patch] Δ(post-patch vs clean) = {(out[0, idx] - out[1, idx]).abs().mean().item():.6f}")
                
    #             # print(f"[After-patch] Δ(patched vs restore) = {(out[1, idx] - vec.to(out.device)).abs().mean().item():.6f}")

                

    #             # Delta check
    #             # delta = (out[:, idx] - before).abs().mean().item()
    #             # print(f"out[:, {idx}].shape = {out[:, idx].shape}, vec.shape = {vec.shape}")
    #             # print(f"[Patch] Δ after applying patch: {delta:.6f} (vec.mean: {vec.mean().item():.6f})")
    #             # print(f"[vec] mean: {vec.mean().item()}, std: {vec.std().item()}, shape: {vec.shape}")
    #             # print(f"[before] mean: {before.mean().item()}, std: {before.std().item()}")


    #             # --- Extra debug: Check if patch is identical to clean ---
    #             # if torch.allclose(before, vec.to(out.device), atol=1e-6):
    #             #     print(f"[WARN] Patch vector for token {idx} is (almost) identical to pre-patch value.")
        
    #     if retainer is not None:
    #         retainer["after_down_proj"] = out.detach().clone()
        
                
        
    #         # print(f"[Retainer] after_down_proj: {retainer['after_down_proj'].abs().mean().item()}")


    
    #     return out



    return forward_patcher


from einops import einsum


# ! just ablating the diagonal ssm isn't enough.
# TODO: figure out how to ablate the shift-SSM or the conv as well
# also, the "attention" visualization is wrong, because it doesn't take the Conv into account
# technically, ssm doesn't pay attention on a particular token, it pays attention to the entire receptive field
def selective_scan_with_mask(
    self, u, delta, A, B, C, D, mask=None, retainer=None, mask_policy="subtract"
):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]
    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)
    Returns:
        output: shape (b, l, d_in)
    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
    """
    # **Extra Arguments**
    # ! mask: dict[target_idx] = [src_idx1, src_idx2, ...]
    # ! retainer: if an empty dictionary is passed, the retention values will be stored there
    # ! mask_policy:
    # !     - "subtract": subtracts the contribution of src from the target
    # !     - "retain": stores the retention/contribution from the source to target (retainer must be passed)

    (b, l, d_in) = u.shape

    if mask is not None:
        assert b == 1, "masking is only supported for batch size 1"
        # TODO: support batch size > 1? let's not overcomplecate things for now

    n = A.shape[1]

    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
    deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

    # print("-" * 60)
    # print(
    #     f"{A.shape=} | {B.shape=} | {C.shape=} | {D.shape=} | {u.shape=} | {delta.shape=}"
    # )
    print(f"{deltaA.shape=} | {deltaB_u.shape=}")

    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    ys = []
    for i in range(l):

        # print(
        #     f"{deltaA[:, i].shape=} | {x.shape=} | {deltaB_u[:, i].shape=} | {C[:, i, :].shape=}"
        # )

        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")

        # print(f"{i} - {y.shape=}")

        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    # print(y.shape)
    # print(f"{y.norm()=}")

    if mask is not None:
        for target_idx, src_idxs in mask.items():
            if retainer != None:
                retainer[target_idx] = {}
            for src_idx in src_idxs:
                assert (
                    src_idx <= target_idx
                ), f"autoregressive LM, {src_idx=} must be <= to {target_idx=}"
                delta_A_src_to_target = torch.prod(
                    deltaA[:, src_idx + 1 : target_idx + 1], dim=1
                )
                delta_B_src = deltaB_u[:, src_idx]

                # print(f"{delta_A_src_to_target.shape=}")
                # print(f"{delta_B_src.shape=}")

                delta_AB_src = delta_A_src_to_target * delta_B_src

                # print(f"{delta_AB_src.shape=}")

                retention_from_src_to_target = einsum(
                    delta_AB_src, C[:, target_idx, :], "b d_in n, b n -> b d_in"
                )

                # print(f"{retention_from_src_to_target.shape=}")

                if retainer != None:
                    retainer[target_idx][src_idx] = retention_from_src_to_target
                if mask_policy == "subtract":
                    # print(
                    #     f"subtracting {src_idx=} from {target_idx=} >> {retention_from_src_to_target.norm()}"
                    # )
                    y[:, target_idx] -= retention_from_src_to_target

    # print(y)
    print(f"{y.norm()=}")

    # # ! if the mask is everything then y at this position should be exactly zero
    if mask is not None:
        for target_idx in range(l):
            print(
                f"||y_{target_idx}|| = {y[:, target_idx].norm().item()} | IS IT ZERO: {torch.allclose(y[:, target_idx], torch.zeros_like(y[:, target_idx]), atol=1e-3)} | max = {y[:, target_idx].max().item()} | min = {y[:, target_idx].min().item()}"
            )
        print("-------------------------------------")

    y = y + u * D

    return y


# Testing code for selective_scan_with_mask

# from src.utils import experiment_utils
# experiment_utils.set_seed(123456)

# u = torch.randn(1, 4, 5120)
# delta = torch.randn(1, 4, 5120)
# A = torch.randn(5120, 16)
# B = torch.randn(1, 4, 16)
# C = torch.randn(1, 4, 16)
# D = torch.randn(5120)

# output = selective_scan_with_mask(
#     self=None,
#     u=u,
#     delta=delta,
#     A=A,
#     B=B,
#     C=C,
#     D=D,
#     # mask = {0:[]}
#     mask={0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]},
# )

# output.shape
