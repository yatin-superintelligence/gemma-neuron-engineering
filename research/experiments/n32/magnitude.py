"""Read-only local edit magnitudes before post-feedforward normalization."""
from contextlib import contextmanager

import torch

from research.experiments.n32.current_site import group_policy


@contextmanager
def observed_group_policy(model, coordinates, operation, donor, sequence_length):
    grouped = {}
    for coordinate in coordinates:
        grouped.setdefault(coordinate['layer'], []).append(coordinate['channel'])
    layers = model.model.language_model.layers
    prior, before_handles = {}, []
    magnitude = {'layers': [], 'read_only': True, 'applied_scale_or_vector': False,
                 'scope': 'Local raw down-projection change, before normalization; not a residual mediation fraction.'}
    try:
        for layer, channels in sorted(grouped.items()):
            def before(module, arguments, layer=layer, channels=channels):
                if layer in prior:
                    raise ValueError('One current observation per native layer required')
                prior[layer] = arguments[0][:, -1, channels].detach().clone()
            before_handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(before))
        with group_policy(model, coordinates, operation, donor, sequence_length) as edit:
            after_handles = []
            try:
                for layer, channels in sorted(grouped.items()):
                    def after(module, arguments, layer=layer, channels=channels):
                        original = prior.pop(layer)
                        actual = arguments[0][:, -1, channels].detach()
                        version = module.weight._version
                        delta = actual.float() - original.float()
                        columns = module.weight[:, channels].detach().float()
                        raw_delta = delta @ columns.T
                        assert module.weight._version == version
                        assert torch.isfinite(raw_delta).all()
                        magnitude['layers'].append({
                            'layer': layer, 'channels': len(channels),
                            'native_dtype': str(actual.dtype), 'raw_projection_arithmetic': 'torch.float32',
                            'before_squared_sum': float(original.double().square().sum()),
                            'after_squared_sum': float(actual.double().square().sum()),
                            'delta_squared_sum': float(delta.double().square().sum()),
                            'raw_down_delta_l2': float(raw_delta.double().square().sum().sqrt()),
                            'down_columns_squared_sum': float(columns.double().square().sum()),
                            'weight_version_unchanged': True})
                        # Returning None observes the already edited input without replacing it.
                    after_handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(after))
                yield {'edit': edit, 'magnitude': magnitude}
                assert not prior and len(magnitude['layers']) == len(grouped)
            finally:
                for handle in after_handles:
                    handle.remove()
    finally:
        for handle in before_handles:
            handle.remove()
        prior.clear()
