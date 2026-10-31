"""Read native MLP activity without changing its forward tensors or parameters."""

import torch


def spaced_positions(length, maximum):
    """Choose deterministic integer endpoints without response-dependent selection."""
    if not isinstance(length, int) or not isinstance(maximum, int):
        raise ValueError('Integer lengths required')
    if min(length, maximum) < 1:
        raise ValueError('Positive lengths required')
    count = min(length, maximum)
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


class NativeActivity:
    """Measure post-gating channels and both sides of post-MLP normalization.

    This observer retains selected token slices and streaming moments. It never
    returns a replacement forward value. Call ``begin`` before each model call
    and ``finish`` afterward. Moments cover only the last processed position of
    each call, avoiding extra weight for long prompt prefills.
    """

    def __init__(self, model, coordinates):
        self.layers = list(model.model.language_model.layers)
        self.coordinates = [tuple(x) for x in coordinates]
        if not self.coordinates or self.coordinates != sorted(set(self.coordinates)):
            raise ValueError('Coordinates must be nonempty, unique and sorted')
        self.widths = [layer.mlp.down_proj.in_features for layer in self.layers]
        for layer, channel in self.coordinates:
            if type(layer) is not int or type(channel) is not int:
                raise ValueError('Integer coordinates required')
            if not 0 <= layer < len(self.layers) or not 0 <= channel < self.widths[layer]:
                raise ValueError('Coordinate outside native MLP')
        self.indices = {
            layer: [channel for li, channel in self.coordinates if li == layer]
            for layer in range(len(self.layers))
        }
        with torch.no_grad():
            self.down_norms = [layer.mlp.down_proj.weight.detach().float().norm(dim=0) for layer in self.layers]
        if not all(torch.isfinite(x).all() for x in self.down_norms):
            raise ValueError('Nonfinite native output columns')
        self.handles = []
        self.active = None
        self.reset_moments()

    def reset_moments(self):
        if self.active is not None:
            raise ValueError('Cannot reset an unfinished forward')
        self.steps = 0
        self.sums = [torch.zeros((3, w), dtype=torch.float64, device=n.device) for w, n in zip(self.widths, self.down_norms)]

    def begin(self, sequence_length, positions, toplist=False):
        if not self.handles or self.active is not None:
            raise ValueError('Observer is closed or a forward is unfinished')
        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError('Positive native sequence length required')
        positions = list(positions)
        if positions != sorted(set(positions)) or any(type(i) is not int or not 0 <= i < sequence_length for i in positions):
            raise ValueError('Invalid sampled positions')
        self.active = {'length': sequence_length, 'positions': positions, 'toplist': bool(toplist), 'h': {}, 'raw': {}, 'norm': {}, 'tops': {}}

    def _check(self, value, layer, key, width):
        frame = self.active
        if frame is None or layer in frame[key]:
            raise ValueError('Missing begin or repeated native module call')
        if value.ndim != 3 or tuple(value.shape) != (1, frame['length'], width):
            raise ValueError('Unexpected native tensor shape')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite observed tensor')
        return frame

    @torch.no_grad()
    def _input(self, layer, module, args):
        value = args[0].detach()
        frame = self._check(value, layer, 'h', self.widths[layer])
        current = value[0, -1].double()
        self.sums[layer][0].add_(current)
        self.sums[layer][1].add_(current.abs())
        self.sums[layer][2].add_(current.square())
        sample = value[0, frame['positions']].float()
        picked = sample[:, self.indices[layer]]
        stats = torch.stack((sample.mean(-1), sample.square().mean(-1).sqrt(), sample.abs().amax(-1)), dim=-1)
        frame['h'][layer] = (picked.cpu(), stats.cpu())
        if frame['toplist']:
            magnitudes = current.float().abs() * self.down_norms[layer]
            order = torch.argsort(magnitudes, descending=True, stable=True)[:min(8, self.widths[layer])]
            frame['tops'][layer] = {
                'channels': order.cpu().tolist(),
                'activation': current[order].float().cpu().tolist(),
                'raw_projection_magnitude': magnitudes[order].cpu().tolist(),
            }

    @torch.no_grad()
    def _output(self, layer, key, module, args, output):
        value = output.detach()
        width = self.layers[layer].mlp.down_proj.out_features
        frame = self._check(value, layer, key, width)
        frame[key][layer] = value[0, frame['positions']].float().norm(dim=-1).cpu()

    def finish(self):
        frame = self.active
        if frame is None:
            raise ValueError('No forward is active')
        expected = set(range(len(self.layers)))
        if any(set(frame[key]) != expected for key in ['h', 'raw', 'norm']):
            raise ValueError('Incomplete native module coverage')
        if frame['toplist'] and set(frame['tops']) != expected:
            raise ValueError('Incomplete native toplist coverage')
        units = torch.cat([frame['h'][i][0] for i in range(len(self.layers))], dim=1)
        stats = torch.stack([
            torch.cat((frame['h'][i][1], frame['raw'][i][:, None], frame['norm'][i][:, None]), dim=-1)
            for i in range(len(self.layers))
        ], dim=1)
        self.active = None
        self.steps += 1
        return {'values': units.numpy(), 'layer_stats': stats.numpy(), 'tops': frame['tops']}

    def moments(self):
        if self.active is not None or self.steps < 1:
            raise ValueError('No complete trajectory moments')
        return torch.cat(self.sums, dim=1) / self.steps

    def __enter__(self):
        if self.handles:
            raise ValueError('Observer already installed')
        try:
            for i, layer in enumerate(self.layers):
                def pre(module, args, i=i):
                    self._input(i, module, args)
                def raw(module, args, output, i=i):
                    self._output(i, 'raw', module, args, output)
                def norm(module, args, output, i=i):
                    self._output(i, 'norm', module, args, output)
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
                self.handles.append(layer.mlp.down_proj.register_forward_hook(raw))
                self.handles.append(layer.post_feedforward_layernorm.register_forward_hook(norm))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        unfinished = self.active is not None
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = None
        if unfinished and exc_type is None:
            raise ValueError('Observer exited before a forward finished')

