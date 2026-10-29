"""Lossless neuron-edit bundles. No training and no full-checkpoint duplication."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import argparse
import hashlib
import json
import torch
from safetensors.torch import load_file
from interpretability.gated_mlp_surgery import (
    DenseGatedMLP, delete_channels, duplicate_channels, add_channels,
    edit_output_vectors, steer_channels,
)
from interpretability.expansion import NewGatedNeurons, insert_branch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def inspect_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'bundle.json').read_text())
    if manifest['schema_version'] != 1 or manifest['optimizer_steps'] != 0:
        raise ValueError('Unsupported bundle or nonzero training steps')
    for name, digest in manifest['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_sha256(path) != digest:
            raise ValueError('Bundle file/hash mismatch: ' + name)
    return manifest


def dense_target(mlp):
    weights = []
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        projection = getattr(mlp, name)
        if getattr(projection.weight, 'quant_state', None) is not None:
            from bitsandbytes.functional import dequantize_4bit
            weight = dequantize_4bit(projection.weight.data, projection.weight.quant_state)
        else:
            weight = projection.weight.detach()
        weights.append(weight.to(dtype=torch.bfloat16))
    return DenseGatedMLP(*weights, mlp.act_fn).eval().requires_grad_(False)


def apply_physical_edit(dense, edit, target, partner, basis):
    """Return modified stored weights, using the released exact vectors where needed."""
    reference = {'gate_target': dense.gate_proj.weight[target],
                 'up_partner': dense.up_proj.weight[partner],
                 'down_target': dense.down_proj.weight[:, target]}
    for name, expected in reference.items():
        if not torch.equal(basis[name].to(expected.device, expected.dtype), expected):
            raise ValueError('Basis does not match the pinned dense target: ' + name)
    if edit == 'dense_delete':
        return delete_channels(dense, [target])
    if edit == 'dense_scale2':
        return edit_output_vectors(dense, [target], basis['down_target'].unsqueeze(1))
    if edit in ['dense_clone_unsplit', 'dense_clone_split']:
        return duplicate_channels(dense, [target], preserve_function=edit == 'dense_clone_split')
    if edit in ['dense_new_zero', 'dense_recombined']:
        down = basis['down_target'].unsqueeze(1)
        down = torch.zeros_like(down) if edit == 'dense_new_zero' else down / 2
        return add_channels(dense, basis['gate_target'].unsqueeze(0),
                            basis['up_partner'].unsqueeze(0), down)
    raise ValueError('Unknown physical edit: ' + edit)


def projection_hashes(mlp):
    return {name: tensor_sha256(getattr(mlp, name).weight)
            for name in ['gate_proj', 'up_proj', 'down_proj']}


def load_n02_base(folder, device='cuda:0', verify_source=True):
    """Load the exact text-only parent configuration and declared representation."""
    import transformers
    from huggingface_hub import hf_hub_download
    folder = Path(folder)
    manifest = inspect_bundle(folder)
    if manifest['kind'] != 'n02_physical_edits':
        raise ValueError('Expected an N02 physical-edit bundle')
    if transformers.__version__ != '5.8.0':
        raise ValueError('The validated loader uses transformers==5.8.0')
    parent, revision = manifest['base']['checkpoint'], manifest['base']['revision']
    if verify_source:
        path = hf_hub_download(parent, filename='model.safetensors', revision=revision)
        if file_sha256(path) != manifest['base']['payload_sha256']:
            raise ValueError('Base checkpoint payload mismatch')
    config = transformers.AutoConfig.from_pretrained(parent, revision=revision, trust_remote_code=False)
    config.vision_config = None
    config.audio_config = None
    kwargs = {}
    if manifest['representation'] == 'nf4_text_same_source':
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig.from_dict(
            json.loads((folder / 'quantization_config.json').read_text()))
    model, info = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        parent, revision=revision, config=config, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={'': device}, attn_implementation='eager',
        output_loading_info=True, low_cpu_mem_usage=True, **kwargs)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Required checkpoint tensors did not load')
    model.eval().requires_grad_(False)
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise ValueError('Expected tied input/output embeddings')
    if model.model.vision_tower is not None or model.model.audio_tower is not None:
        raise ValueError('Expected text-only modules')
    target = model.model.language_model.layers[manifest['layer']].mlp
    if projection_hashes(target) != manifest['original_projection_hashes']:
        raise ValueError('Original target-layer weight bytes differ')
    return model, manifest


def reconstruct_n02(folder, edit, device='cuda:0', verify_source=True):
    """The returned model keeps the physical edit applied. Rebuild using the bundle."""
    folder = Path(folder)
    model, manifest = load_n02_base(folder, device=device, verify_source=verify_source)
    if edit not in manifest['edits']:
        raise ValueError('Edit is not declared in this bundle')
    layer = model.model.language_model.layers[manifest['layer']]
    dense = dense_target(layer.mlp)
    basis = load_file(str(folder / 'basis.safetensors'))
    edited = apply_physical_edit(dense, edit, manifest['target'], manifest['partner'], basis)
    actual = projection_hashes(edited)
    expected = manifest['expected_edited_projection_hashes'].get(edit)
    if expected is not None and actual != expected:
        raise ValueError('Reconstructed edit weight bytes differ from the recorded model')
    layer.mlp = edited
    return model, {'bundle': str(folder), 'edit': edit, 'projection_hashes': actual,
                   'expected_hashes_available': expected is not None,
                   'base_revision': manifest['base']['revision'], 'optimizer_steps': 0}


def load_n01_base(folder, device='cuda:0'):
    import transformers
    manifest = inspect_bundle(folder)
    if manifest['kind'] != 'n01_branches_and_gain_policies':
        raise ValueError('Expected an N01 branch/policy bundle')
    if transformers.__version__ != '5.8.0':
        raise ValueError('The validated loader uses transformers==5.8.0')
    parent, revision = manifest['base']['checkpoint'], manifest['base']['revision']
    config = transformers.AutoConfig.from_pretrained(parent, revision=revision, trust_remote_code=False)
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        parent, revision=revision, trust_remote_code=False, device_map={'': device},
        dtype=torch.bfloat16, quantization_config=quant, attn_implementation='eager', low_cpu_mem_usage=True)
    # Match the frozen N01 vector conversion; embedding tables keep their load dtype.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.__class__.__name__ != 'Params4bit' and parameter.is_floating_point() and parameter.ndim == 1:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_disable()
    return model.eval(), manifest


@contextmanager
def apply_n01_branch(model, folder, branch_name):
    folder = Path(folder)
    manifest = inspect_bundle(folder)
    config = manifest['branches'][branch_name]
    device = next(model.parameters()).device
    branch = NewGatedNeurons(config['hidden_size'], config['width'], seed=config['seed'],
                            normalize=config['normalize'], device=device).eval().requires_grad_(False)
    branch.load_state_dict(load_file(str(folder / config['file']), device=str(device)))
    with insert_branch(model.get_submodule(config['target_module']), branch, config['placement']) as receipt:
        yield receipt


@contextmanager
def apply_n01_gain(model, folder, group_name, gain):
    manifest = inspect_bundle(folder)
    if group_name not in manifest['steering_groups'] or gain not in manifest['allowed_gains_by_group'][group_name]:
        raise ValueError('Undeclared group or gain')
    groups = {}
    for coordinate in manifest['steering_groups'][group_name]:
        groups.setdefault(coordinate['layer'], []).append(coordinate['channel'])
    with ExitStack() as stack:
        audits = [stack.enter_context(steer_channels(model.get_submodule(
                  f'model.language_model.layers.{layer}.mlp'), channels, gain=gain, token_scope='last'))
                  for layer, channels in groups.items()]
        yield audits


def main():
    parser = argparse.ArgumentParser(description='Inspect or reconstruct a saved neuron-edit bundle.')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--edit')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rebuild', action='store_true', help='Download/load the pinned base and reconstruct on the chosen device.')
    args = parser.parse_args()
    manifest = inspect_bundle(args.bundle)
    if not args.rebuild:
        print(json.dumps({'bundle_valid': True, 'kind': manifest['kind'], 'base': manifest['base'],
                          'optimizer_steps': 0, 'base_weights_downloaded': False}, indent=2))
        return
    if not args.edit:
        parser.error('--edit is required with --rebuild')
    model, receipt = reconstruct_n02(args.bundle, args.edit, device=args.device)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
