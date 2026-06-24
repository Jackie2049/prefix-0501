import argparse
import os
import stat

import torch


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir", help="llama model dir")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir", help="output model dir")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=1,
                        help="should be consistent with megatron")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument("--added-token-num", type=int, default=0, help="the number of added tokens")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="num layers")
    parser.add_argument('--swiglu', action='store_true',
                        help='Use gated linear units and SiLU activation instead of default gelu')
    return parser.parse_args()


def check_divisible(denominator, molecule, error_info=None):
    if denominator % molecule == 0:
        return
    raise ValueError(f"{denominator} is not divisible by {molecule}. {error_info}")


def check_equal(tensor_a, tensor_b, error_info=None):
    if tensor_a == tensor_b:
        return
    raise ValueError(f"{tensor_a} is not equal to {tensor_b}. {error_info}")


def vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tp):
    # Pad vocab size so it is divisible by model parallel size and still having GPU friendly size.
    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tp
    while (after % multiple) != 0:
        after += 1
    return after


def pad_embed(w, make_vocab_size_divisible_by, tp, added_token_num):
    padded_size = vocab_size_with_padding(w.shape[0] + added_token_num, make_vocab_size_divisible_by, tp)
    if padded_size == w.shape[0]:
        return w.clone()
    return torch.cat([w, w[-(padded_size - w.shape[0]):, ...]], dim=0)


args = get_args()
input_model_dir = args.input_model_dir


def get_emb_w_param(model, args, tp_size, tp_rank):
    emb_w = model['language_model']['embedding']['word_embeddings']['weight']
    emb_w = pad_embed(emb_w, args.make_vocab_size_divisible_by, tp_size, args.added_token_num)

    return {'word_embeddings': {'weight': torch.chunk(emb_w, tp_size, dim=0)[tp_rank]}}


def get_lm_w_param(model, args, tp_size, tp_rank):
    lm_w = model['language_model']['output_layer']['weight']
    lm_w = pad_embed(lm_w, args.make_vocab_size_divisible_by, tp_size, args.added_token_num)
    return {'weight': torch.chunk(lm_w, tp_size, dim=0)[tp_rank]}


def transformer_param_cvt(model, rank_model, tp_size, tp_rank, ori_i, pp_i, is_swiglu):
    input_norm_w = model['language_model']['encoder'][f'layers.{ori_i}.input_norm.weight']
    rank_model['language_model']['encoder'][f'layers.{pp_i}.input_norm.weight'] = input_norm_w.clone()

    qkv_w = model['language_model']['encoder'][f'layers.{ori_i}.self_attention.query_key_value.weight']
    rank_qkv_w = torch.chunk(qkv_w, tp_size, dim=0)[tp_rank].clone()
    rank_model['language_model']['encoder'][
        f'layers.{pp_i}.self_attention.query_key_value.weight'] = rank_qkv_w

    dense_w = model['language_model']['encoder'][f'layers.{ori_i}.self_attention.dense.weight']
    rank_dense_w = torch.chunk(dense_w, tp_size, dim=1)[tp_rank].clone()
    rank_model['language_model']['encoder'][f'layers.{pp_i}.self_attention.dense.weight'] = rank_dense_w

    post_norm_w = model['language_model']['encoder'][f'layers.{ori_i}.post_attention_norm.weight']
    rank_model['language_model']['encoder'][
        f'layers.{pp_i}.post_attention_norm.weight'] = post_norm_w.clone()

    h_to_4h = model['language_model']['encoder'][f'layers.{ori_i}.mlp.dense_h_to_4h.weight']
    if is_swiglu:
        h_to_4h_arr = torch.chunk(h_to_4h, 2, dim=0)
        up_w = h_to_4h_arr[0]
        gate_w = h_to_4h_arr[1]
        rank_up_w = torch.chunk(up_w, tp_size, dim=0)[tp_rank].clone()
        rank_gate_w = torch.chunk(gate_w, tp_size, dim=0)[tp_rank].clone()
        rank_h_to_4h = torch.cat([rank_up_w, rank_gate_w], dim=0)
        rank_model['language_model']['encoder'][f'layers.{pp_i}.mlp.dense_h_to_4h.weight'] = rank_h_to_4h
    else:
        up_w = model['language_model']['encoder'][f'layers.{ori_i}.mlp.dense_h_to_4h.weight']
        rank_up_w = torch.chunk(up_w, tp_size, dim=0)[tp_rank].clone()
        rank_model['language_model']['encoder'][f'layers.{pp_i}.mlp.dense_h_to_4h.weight'] = rank_up_w

    down_w = model['language_model']['encoder'][f'layers.{ori_i}.mlp.dense_4h_to_h.weight']
    rank_down_w = torch.chunk(down_w, tp_size, dim=1)[tp_rank].clone()
    rank_model['language_model']['encoder'][f'layers.{pp_i}.mlp.dense_4h_to_h.weight'] = rank_down_w


def no_vpp_param_cvt(model, args, ckpt_name):
    output_model_dir = args.output_model_dir
    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    n_layer = args.num_layers
    is_swiglu = args.swiglu

    pp_n_layer = n_layer // pp_size
    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            rank_model = {
                "language_model": {
                    "encoder": {}
                }
            }

            if pp_rank == 0:
                rank_model['language_model']['embedding'] = get_emb_w_param(model, args, tp_size, tp_rank)

            if pp_rank == pp_size - 1:
                final_norm_w = model['language_model']['encoder']['final_norm.weight']
                rank_model['language_model']['encoder']['final_norm.weight'] = final_norm_w.clone()

                rank_model['language_model']['output_layer'] = get_lm_w_param(model, args, tp_size, tp_rank)

            for pp_i in range(pp_n_layer):
                ori_i = pp_n_layer * pp_rank + pp_i
                transformer_param_cvt(model, rank_model, tp_size, tp_rank, ori_i, pp_i, is_swiglu=is_swiglu)

            iteration = '0' if ckpt_name == 'release' else int(ckpt_name)
            model_dict = {"checkpoint_version": 3.0, 'model': rank_model, 'iteration': iteration}

            sub_dir = "release" if ckpt_name == "release" else f"iter_{str(ckpt_name).zfill(7)}"
            save_dir = os.path.join(output_model_dir, sub_dir, f"mp_rank_{str(tp_rank).zfill(2)}") \
                if pp_size == 1 else os.path.join(output_model_dir, sub_dir,
                                                  f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model_dict, os.path.join(save_dir, "model_optim_rng.pt"))


def vpp_param_cvt(model, args, ckpt_name):
    output_model_dir = args.output_model_dir
    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    vp_stage = args.num_layers_per_virtual_pipeline_stage
    n_layer = args.num_layers
    is_swiglu = args.swiglu

    pp_n_layer = n_layer // pp_size
    vpp_size = pp_n_layer // vp_stage

    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            model_dict = {"checkpoint_version": 3.0}
            for vp_rank in range(vpp_size):
                rank_model = {
                    "language_model": {
                        "encoder": {}
                    }
                }
                if pp_rank == 0 and vp_rank == 0:
                    rank_model['language_model']['embedding'] = get_emb_w_param(model, args, tp_size, tp_rank)

                if pp_rank == pp_size - 1 and vp_rank == vpp_size - 1:
                    final_norm_w = model['language_model']['encoder']['final_norm.weight']
                    rank_model['language_model']['encoder']['final_norm.weight'] = final_norm_w.clone()

                    rank_model['language_model']['output_layer'] = get_lm_w_param(model, args, tp_size, tp_rank)

                for pp_i in range(vp_stage):
                    ori_i = vp_rank * (vp_stage * pp_size) + pp_rank * vp_stage + pp_i
                    transformer_param_cvt(model, rank_model, tp_size, tp_rank, ori_i, pp_i, is_swiglu=is_swiglu)

                model_dict[f"model{vp_rank}"] = rank_model
            sub_dir = "release" if ckpt_name == "release" else f"iter_{str(ckpt_name).zfill(7)}"
            save_dir = os.path.join(output_model_dir, sub_dir,
                                    f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model_dict, os.path.join(save_dir, "model_optim_rng.pt"))


if __name__ == "__main__":
    ckpt_file = os.path.join(input_model_dir, "latest_checkpointed_iteration.txt")
    if not os.path.exists(ckpt_file):
        raise ValueError(f"not found latest_checkpointed_iteration.txt in input_model_dir: {input_model_dir}")

    with open(ckpt_file, 'r') as f:
        ckpt_name = str(f.read()).strip()

    if ckpt_name == "release":
        model_path = os.path.join(input_model_dir, "release/mp_rank_00/model_optim_rng.pt")
    else:
        try:
            ckpt_name = int(ckpt_name)
            model_path = os.path.join(input_model_dir,
                                      f"iter_{str(ckpt_name).zfill(7)}/mp_rank_00/model_optim_rng.pt")
        except ValueError as _:
            raise ValueError(f"error data in latest_checkpointed_iteration.txt : {ckpt_name}")

    if not os.path.exists(model_path):
        raise ValueError(f"model_path is not exists. {model_path}")

    model_info = torch.load(model_path, map_location='cpu')
    model = model_info['model']

    output_model_dir = args.output_model_dir

    if args.num_layers_per_virtual_pipeline_stage is None:
        no_vpp_param_cvt(model, args, ckpt_name)
    else:
        vpp_param_cvt(model, args, ckpt_name)

    filename = "latest_checkpointed_iteration.txt"
    flags = os.O_RDWR | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP
    print(f"ckpt name : {ckpt_name}")
    with os.fdopen(os.open(os.path.join(output_model_dir, filename), flags, modes),
                   'w') as fout:
        fout.write(f"{ckpt_name}\n")
