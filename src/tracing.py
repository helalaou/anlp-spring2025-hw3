import copy
import json
import logging
import types
from collections import defaultdict
from typing import Literal, Optional, get_args

import baukit
import numpy
import torch
from tqdm import tqdm

import src.tokens as tokenizer_utils

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as Mamba
from mamba_minimal.model import Mamba
from src import dataset, functional
from src.functional import (
    decode_tokens,
    find_token_range,
    make_inputs,
    predict_from_input,
)
from src.hooking.mamba import MambaBlock_Hook_Points, MambaBlockForwardPatcher, Mamba2BlockForwardPatcher
from src.models import ModelandTokenizer

logger = logging.getLogger(__name__)
printed_patch_function = False


def trace_with_patch(
    mt,
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    mamba_block_hook: Optional[
        MambaBlock_Hook_Points
    ] = None,  # what to patch in the corrupted run in the MambaBlock. If None => Patch the whole residual block (Not MambaBlock output)
    alt_subj_patching: bool = False,  # If True, will assume inp shape to be (2, L). Uncorrupted activations with inp[0] will be patched in the run with inp[1]. Will not corrupt the embeddings
):
    # print(f"{alt_subj_patching=}")
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    PatchFunction = MambaBlockForwardPatcher
    if mamba_block_hook is not None:
        assert isinstance(
            mt.model, Mamba
        )or getattr(mt, "is_mamba2", False), "if `mamba_block_hook` is not None, the model should be a Mamba"
        assert mamba_block_hook in get_args(
            MambaBlock_Hook_Points
        ), f"Not a valid MambaBock hook `{mamba_block_hook=}`"
        
                # Select the patcher based on model type
        if getattr(mt, "is_mamba2", False):
            PatchFunction = Mamba2BlockForwardPatcher
        else:
            PatchFunction = MambaBlockForwardPatcher
    
        
    embed_layername = mt.embedder_name

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise
    
    
    def patch_rep(repr, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None and alt_subj_patching == False:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(repr.shape[0] - 1, e - b, repr.shape[2]))
                ).to(repr.device)
                if replace:
                    repr[1:, b:e] = noise_data
                else:
                    repr[1:, b:e] += noise_data
            return repr
        if isinstance(mt.model, Mamba) and mamba_block_hook is not None:
            # don't do anything on MambaBlock if not embedding. The MambaBlockForwardPatcher will take care of it
            return repr
        if layer not in patch_spec:
            return repr
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(repr)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]

        return repr

    mt.reset_forward()  # reset the model to use default forward functions
    additional_layers = [] if trace_layers is None else trace_layers
    if (isinstance(mt.model, Mamba) == False and getattr(mt, "is_mamba2", False) is False ) or (
        (isinstance(mt.model, Mamba) == True or getattr(mt, "is_mamba2", False)) and mamba_block_hook is None
    ):
        # With the patching rules defined, run the patched model in inference.
        with torch.no_grad(), baukit.TraceDict(
            mt.model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
        ) as td:
            outputs_exp = (
                mt.model(input_ids=inp["input_ids"])
                if isinstance(mt.model, Mamba)
                else mt.model(**inp)
            )
    else:
        # uncorrupted run
        patch_layers = list(patch_spec.keys()) + additional_layers
        uncorrupted_activations = {layer: {} for layer in patch_layers}

        for layer in patch_layers:
            block = baukit.get_module(
                mt.model, name=layer + ".mixer"
            )  # MambaBlock naming format
            block.forward = types.MethodType(
                PatchFunction(
                    retainer=uncorrupted_activations[layer],
                ),  # get everything for the uncorrupted run
                block,
            )

        with torch.inference_mode():
            mt.model(
                input_ids=inp["input_ids"][0][None]
            )  # only the first input for the uncorrputed run
            

        # ------------------------------------------------------
        # Corrupted run with patching
        mt.reset_forward()  # reset the model to use default forward functions

        for layer in patch_layers:
            block = baukit.get_module(mt.model, name=layer + ".mixer")


            # cur_patch_spec = {
            #     t: uncorrupted_activations[layer][mamba_block_hook][0, t]
            #     .to(dtype=block.in_proj.weight.dtype, device=block.in_proj.weight.device)
            #     for t in patch_spec[layer]
            # }
            cur_patch_spec = {
                t: uncorrupted_activations[layer][mamba_block_hook][0, t]
                for t in patch_spec[layer]
            }
            

            block.forward = types.MethodType(
                PatchFunction(
                    patch_spec=cur_patch_spec,
                    patch_hook=mamba_block_hook,
                ),
                block,
            )
        

        with torch.inference_mode(), baukit.TraceDict(
            mt.model,
            [embed_layername],
            edit_output=patch_rep,  # passing to patch_rep to noise the embeddings only. Restoring the states is done in the MambaBlockForwardPatcher forwards
        ):
            outputs_exp = mt.model(input_ids=inp["input_ids"])
            
        # ------------------------------------------------------
        mt.reset_forward()  # reset the model to use default forward functions

    # We report softmax probabilities for the answers_t token predictions of interest.
    logits = outputs_exp.logits if hasattr(outputs_exp, "logits") else outputs_exp
    
    # with torch.inference_mode():
    #         outputs_corrupt = mt.model(input_ids=inp["input_ids"][1:].contiguous())
    #         logits_corrupt = outputs_corrupt.logits if hasattr(outputs_corrupt, "logits") else outputs_corrupt
            
    #         print("Logits patched vs corrupted:", (logits_corrupt[:, -1] - logits[1:, -1]).abs().mean().item())
            
            
            
    probs = torch.softmax(logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    
    # print("Logits clean vs patched:", (logits[0, -1] - logits[1, -1]).abs().mean().item())
    # print("Patched prob:", probs.item())
    # print("Predicted token (patched):", logits[1, -1].argmax().item())

    
    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def replace_eos_with_pad(tokenizer, token_list, pad_token="[PAD]"):
    if hasattr(tokenizer, "eos_token") == False:
        return token_list
    for i, token in enumerate(token_list):
        if token == tokenizer.eos_token:
            token_list[i] = pad_token
    return token_list


def calculate_hidden_flow(
    mt: ModelandTokenizer,
    prompt: str,
    subject: str,
    alt_subject: Optional[str] = None,
    num_samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    mamba_block_hook: Optional[MambaBlock_Hook_Points] = None,
):
    
    # print("In Calculate Hidden Flow")
    if not getattr(mt, "is_mamba2", False):
        
        # check appropriate `kind` of module to trace based on the model
        if isinstance(mt.model, Mamba):
            assert kind in ["mlp", "ssm", None]
        else:
            assert kind in ["mlp", "attn", None]

    if alt_subject is None:
        inp = make_inputs(mt.tokenizer, [prompt] * (num_samples + 1))
        with torch.no_grad():
            answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
        e_range = find_token_range(
            string=prompt,
            substring=subject,
            tokenizer=mt.tokenizer,
        )
        if token_range == "subject_last":
            token_range = [e_range[1] - 1]
        elif token_range is not None:
            raise ValueError(f"Unknown token_range: {token_range}")
        
        low_score = trace_with_patch(
            mt,
            inp,
            [],
            answer_t,
            e_range,
            noise=noise,
            uniform_noise=uniform_noise,
            mamba_block_hook=None,  # don't need to patch for calculating the low score
            alt_subj_patching=alt_subject is not None,
        ).item()
    else:
        if "{}" in prompt:
            prompt = prompt.format(subject)
        clean_prompt = prompt
        alt_prompt = prompt.replace(subject, alt_subject)
        with tokenizer_utils.set_padding_side(mt.tokenizer, padding_side="left"):
            inp = mt.tokenizer(
                [clean_prompt, alt_prompt],
                return_tensors="pt",
                padding="longest",
                return_offsets_mapping=True,
            ).to(mt.device)
        offset_mapping = inp.pop("offset_mapping")
        subject_range = find_token_range(
            string=clean_prompt,
            substring=subject,
            tokenizer=mt.tokenizer,
            offset_mapping=offset_mapping[0],
        )
        alt_subj_range = find_token_range(
            string=alt_prompt,
            substring=alt_subject,
            tokenizer=mt.tokenizer,
            offset_mapping=offset_mapping[1],
        )
        assert subject_range[1] == alt_subj_range[1]
        e_range = (min(subject_range[0], alt_subj_range[0]), subject_range[1])
        
        # print("After Preparation, Now Move to Inference")
        

        with torch.no_grad():
            outputs = mt(**inp)
        logits = outputs.logits[:, -1] if hasattr(outputs, "logits") else outputs[:, -1]
        next_token_probs = logits.float().softmax(dim=-1)
        answer_t = next_token_probs[0].argmax(dim=-1)
        base_score = next_token_probs[0, answer_t]  # p(ans|subj)
        low_score = next_token_probs[1, answer_t]  # p(ans|alt_subj)

    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)

        
    # print(f"After Inference, kind is {kind} and mamba_block_hook is {mamba_block_hook}")
    if not kind and not mamba_block_hook:
        
        print("Move to trace_important_states")
        differences = trace_important_states(
            mt,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
            mamba_block_hook=mamba_block_hook,
            alt_subj_patching=alt_subject is not None,
        )
    else:
        if getattr(mt, "is_mamba2", False) or isinstance(mt.model, Mamba):
            module_name_format = mt.layer_name_format
        else:
            module_name_format = (
                mt.mlp_module_name_format
                if kind == "mlp"
                else mt.attn_module_name_format
            )
        

        differences = trace_important_window(
            mt,
            module_name_format,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            token_range=token_range,
            mamba_block_hook=mamba_block_hook,
            alt_subj_patching=alt_subject is not None,
        )
    differences = differences.detach().cpu()
    indirect_effect = dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=replace_eos_with_pad(
            mt.tokenizer, list(decode_tokens(mt.tokenizer, inp["input_ids"][0]))
        ),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=(kind or mamba_block_hook) or "",
    )

    if alt_subject is not None:
        indirect_effect["alt_subject"] = replace_eos_with_pad(
            mt.tokenizer,
            list(
                decode_tokens(
                    mt.tokenizer, inp["input_ids"][1, e_range[0] : e_range[1]]
                )
            ),
        )

    return indirect_effect


def trace_important_states(
    mt,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    mamba_block_hook: Optional[MambaBlock_Hook_Points] = None,
    alt_subj_patching: bool = False,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    
    
    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        row = []
        for layer in range(mt.n_layer):
            r = trace_with_patch(
                mt,
                inp,
                [(tnum, mt.layer_name_format.format(layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                mamba_block_hook=mamba_block_hook,
                alt_subj_patching=alt_subj_patching,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    mt,
    module_name_format,
    inp,
    e_range,
    answer_t,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    mamba_block_hook: Optional[MambaBlock_Hook_Points] = None,
    alt_subj_patching: bool = False,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    
    
    # print(f"In trace_important_window, Looping through token range {token_range} and layers {mt.n_layer}")


    if token_range is None:
        token_range = range(ntoks)
    

    
    for tnum in token_range:
        row = []
        for layer in range(mt.n_layer):
            layerlist = [
                (tnum, module_name_format.format(L))
                for L in range(
                    max(0, layer - window // 2), min(mt.n_layer, layer - (-window // 2))
                )
            ]
            
            
            r = trace_with_patch(
                mt,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                mamba_block_hook=mamba_block_hook,
                alt_subj_patching=alt_subj_patching,
            )
            row.append(r)
            
        table.append(torch.stack(row))
    
    return torch.stack(table)


from src.dataset.dataclasses import (
    PredictedToken,
    RelationSample,
    ReprReplacementResults,
)
from src.functional import get_h, patch_repr, predict_next_token
from src.models import ModelandTokenizer

# def trace_with_patching_from_alt_subj(
#     mt,
#     inp,  # A set of inputs (2, Num Tokens)
#     states_to_patch,  # A list of (token index, layername) triples to restore
#     answers_t,  # Answer probabilities to collect
#     tokens_to_mix,  # Range of tokens to corrupt (begin, end)
#     noise=0.1,  # Level of noise to add
#     uniform_noise=False,
#     replace=False,  # True to replace with instead of add noise
#     trace_layers=None,  # List of traced outputs to return
#     mamba_block_hook: Optional[
#         MambaBlock_Hook_Points
#     ] = None,  # what to patch in the corrupted run in the MambaBlock. If None => Patch the whole residual block (Not MambaBlock output)
# ):
#     assert (
#         inp.dim(0) == 2
#     ), "Need to pass 2 inputs for the uncorrupted and corrupted runs"


def patch_individual_layers_for_single_edit(
    mt: ModelandTokenizer,
    layers: list[int],
    orig_sample: RelationSample,
    edit_sample: RelationSample,
    query: str,
) -> ReprReplacementResults:
    # TODO: Support for multiple edits
    # ! Multiple edit acting weird. Could not find the bug.

    if "{}" in query:
        query = query.format(orig_sample.subject)

    edit_h = get_h(
        mt=mt,
        prompt=query.replace(orig_sample.subject, edit_sample.subject),
        subject=edit_sample.subject,
        layers=[mt.layer_name_format.format(layer_idx) for layer_idx in layers],
    )

    tokenized = mt.tokenizer(
        query, return_offsets_mapping=True, return_tensors="pt"
    ).to(mt.device)
    offset_mapping = tokenized.pop("offset_mapping")[0]

    subject_start, subject_end = find_token_range(
        query,
        orig_sample.subject,
        tokenizer=mt.tokenizer,
        offset_mapping=offset_mapping,
    )

    subj_last_idx = subject_end - 1
    edit_rank_after_patching: dict[int, tuple[int, PredictedToken]] = {}
    predictions: dict[int, list[PredictedToken]] = {}
    edit_token = mt.tokenizer.decode(tokenized["input_ids"][0][subj_last_idx])

    logger.debug("=" * 100)
    logger.debug(
        f"({orig_sample.subject}, {orig_sample.object}) => ({edit_sample.subject}, {edit_sample.object}) | edit_idx={subj_last_idx}[{edit_token}]"
    )

    for layer_idx in layers:
        layer_name = mt.layer_name_format.format(layer_idx)
        with baukit.Trace(
            module=mt.model,
            layer=layer_name,
            edit_output=patch_repr(
                patch_layer=layer_name,
                patch_idx=subj_last_idx,
                patch_vector=edit_h[layer_name],
            ),
        ):
            preds, edit_answer_rank = predict_next_token(
                mt=mt,
                prompt=query,
                token_of_interest=(
                    f" {edit_sample.object}"
                    if "llama" not in mt.model_name.lower()
                    else edit_sample.object
                ),  # because LLaMA tokenizers handle spacing dynamically
            )
        predictions[layer_idx] = preds[0]
        edit_rank_after_patching[layer_idx] = edit_answer_rank[0]
        logger.debug(
            f"Layer {layer_idx} => rank({edit_sample.object})={edit_answer_rank[0][0]} [{edit_answer_rank[0][1]}]  | preds={', '.join(str(p) for p in preds[0])}"
        )
    logger.debug("-" * 100)

    return ReprReplacementResults(
        source_QA=orig_sample,
        edit_QA=edit_sample,
        edit_index=subj_last_idx,
        edit_token=mt.tokenizer.decode(tokenized["input_ids"][0][subj_last_idx]),
        predictions_after_patching=predictions,
        rank_edit_ans_after_patching=edit_rank_after_patching,
    )


def detensorize_indirect_effects(indirect_effects):
    hf = copy.deepcopy(indirect_effects)
    for k in hf:
        if isinstance(hf[k], torch.Tensor):
            hf[k] = hf[k].item() if k == "high_score" else hf[k].cpu().numpy().tolist()
    return hf


def load_causal_traces(file="causal_traces.json"):
    with open(file, "r") as f:
        indirect_effect_collection = json.load(f)

    for subject in indirect_effect_collection:
        for key in indirect_effect_collection[subject]:
            if (
                isinstance(indirect_effect_collection[subject][key], list)
                and isinstance(indirect_effect_collection[subject][key][0], str)
                == False
            ):
                indirect_effect_collection[subject][key] = torch.tensor(
                    indirect_effect_collection[subject][key]
                )

    return indirect_effect_collection


def calculate_average_indirect_effects(
    mt: ModelandTokenizer,
    prompt: str,
    samples: list[RelationSample],
    corruption_strategy: Literal["corrupt", "alt_patch"] = "alt_patch",
    n_trials: int | None = None,
    save_path: str | None = None,
    verbose: bool = False,
    **kwargs,
):
    indirect_effect_collection = {}
    if corruption_strategy == "alt_patch":
        edit_targets = functional.random_edit_targets(samples=samples)
    samples = samples if n_trials is None else samples[:n_trials]
    for sample in tqdm(samples):
        alt_subject = None
        if corruption_strategy == "alt_patch":
            alt_subject = edit_targets[sample].subject

        if verbose:
            logger.debug(
                f"tracing for {sample.subject} => {alt_subject if alt_subject else '*corrupted*'} state"
            )
        indirect_effects = calculate_hidden_flow(
            mt=mt,
            prompt=prompt,
            subject=sample.subject,
            alt_subject=alt_subject,
            **kwargs,
        )
        indirect_effect_collection[sample.subject] = detensorize_indirect_effects(
            indirect_effects
        )

        if save_path is not None:
            # make sure save after each trial
            with open(save_path, "w") as f:
                json.dump(indirect_effect_collection, f)

    return average_indirect_effects(indirect_effect_collection)


def average_indirect_effects(indirect_effect_collection, relative_recovery=False):
    aie = {
        "input_tokens": [
            "prefix",
            "subject_[:-2]",
            "subject_2nd_last",
            "subject_last",
            "further tokens",
            "last token",
        ],
        "answer": "answer",
    }

    prefixes = []
    low_scores = []
    subject_first = []
    subject_2nd_last = []
    subject_last = []
    further_tokens = []
    last_token = []
    for subject in indirect_effect_collection:
        result = indirect_effect_collection[subject]
        differences = torch.tensor(result["scores"])
        if relative_recovery:
            differences = differences / (result["high_score"] - result["low_score"])
        prefixes.append(differences[: result["subject_range"][0]].mean(dim=0))
        low_scores.append(result["low_score"])
        subject_last.append(differences[result["subject_range"][1] - 1])
        subject_2nd_last.append(differences[result["subject_range"][1] - 2])
        #     print(result['subject_range'],  result['subject_range'][1] - 2, differences.shape)
        if result["subject_range"][1] - 2 != result["subject_range"][0]:
            subject_first.append(
                differences[
                    result["subject_range"][0] : result["subject_range"][1] - 2
                ].mean(dim=0)
            )
        else:
            subject_first.append(torch.zeros(differences.shape[1]))
        last_token.append(differences[-1])

        # print(result["subject_range"][1], len(result["input_tokens"]))
        if result["subject_range"][1] != len(result["input_tokens"]) - 1:
            further_tokens.append(
                differences[result["subject_range"][1] : -1].mean(dim=0)
            )
        else:
            further_tokens.append(torch.zeros(differences.shape[1]))

    prefixes = torch.stack(prefixes).mean(dim=0)
    subject_first = torch.stack(subject_first).mean(dim=0)
    subject_2nd_last = torch.stack(subject_2nd_last).mean(dim=0)
    subject_last = torch.stack(subject_last).mean(dim=0)
    further_tokens = torch.stack(further_tokens).mean(dim=0)
    last_token = torch.stack(last_token).mean(dim=0)

    aie["low_score"] = torch.tensor(low_scores).mean()
    aie["kind"] = indirect_effect_collection[subject]["kind"]
    aie["subject_range"] = (1, 4)

    aie["scores"] = torch.stack(
        [
            prefixes,
            subject_first,
            subject_2nd_last,
            subject_last,
            further_tokens,
            last_token,
        ]
    )

    return aie


# def trace_with_repatch(
#     model,
#     embed_layername,  # The model
#     inp,  # A set of inputs
#     states_to_patch,  # A list of (token index, layername) triples to restore
#     states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
#     answers_t,  # Answer probabilities to collect
#     tokens_to_mix,  # Range of tokens to corrupt (begin, end)
#     noise=0.1,  # Level of noise to add
#     uniform_noise=False,
# ):
#     rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
#     if uniform_noise:
#         prng = lambda *shape: rs.uniform(-1, 1, shape)
#     else:
#         prng = lambda *shape: rs.randn(*shape)
#     patch_spec = defaultdict(list)
#     for t, l in states_to_patch:
#         patch_spec[l].append(t)
#     unpatch_spec = defaultdict(list)
#     for t, l in states_to_unpatch:
#         unpatch_spec[l].append(t)

#     def untuple(x):
#         return x[0] if isinstance(x, tuple) else x

#     # Define the model-patching rule.
#     def patch_rep(x, layer):
#         if layer == embed_layername:
#             # If requested, we corrupt a range of token embeddings on batch items x[1:]
#             if tokens_to_mix is not None:
#                 b, e = tokens_to_mix
#                 x[1:, b:e] += noise * torch.from_numpy(
#                     prng(x.shape[0] - 1, e - b, x.shape[2])
#                 ).to(x.device)
#             return x
#         if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
#             return x
#         # If this layer is in the patch_spec, restore the uncorrupted hidden state
#         # for selected tokens.
#         h = untuple(x)
#         for t in patch_spec.get(layer, []):
#             h[1:, t] = h[0, t]
#         for t in unpatch_spec.get(layer, []):
#             h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
#         return x

#     # With the patching rules defined, run the patched model in inference.
#     for first_pass in [True, False] if states_to_unpatch else [False]:
#         with torch.no_grad(), nethook.TraceDict(
#             model,
#             [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
#             edit_output=patch_rep,
#         ) as td:
#             outputs_exp = model(**inp)
#             if first_pass:
#                 first_pass_trace = td

#     # We report softmax probabilities for the answers_t token predictions of interest.
#     probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

#     return probs
