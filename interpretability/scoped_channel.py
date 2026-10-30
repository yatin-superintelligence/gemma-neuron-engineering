"""Read-only scope witnesses around prescribed native scalar interventions."""
from contextlib import contextmanager
import hashlib
import torch
from interpretability.anchor_intervention import anchor_intervention


def digest(value):
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def bit_equal(left,right):
    return torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


@contextmanager
def scoped_channel(model,coordinates,anchor,operation='gain',gain=1,donor=None):
    """Observe live before/after tensors without retaining full activation dumps.

    A caller may install a full-site background first. The scope comparison
    then uses that live local input rather than an unmodified earlier run.
    Cached single-token steps must skip the anchor intervention.
    """
    grouped={}
    for coordinate in coordinates:grouped.setdefault(coordinate['layer'],[]).append(coordinate)
    layers=model.model.language_model.layers;handles=[];pending={}
    witness={'prefill_checks':0,'cached_checks':0,'all_other_values_bit_exact':True,'all_targets_bit_exact':True,'prefills':[]}
    try:
        for layer in grouped:
            def before(module,args,layer=layer):
                assert layer not in pending
                pending[layer]=args[0].detach().clone()
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(before))
        with anchor_intervention(model,coordinates,anchor,operation=operation,gain=gain,donor=donor)as edit:
            after_handles=[]
            try:
                for layer,cs in grouped.items():
                    channels=[c['channel']for c in cs]
                    def after(module,args,layer=layer,cs=cs,channels=channels):
                        before=pending.pop(layer);changed=args[0]
                        assert changed.shape==before.shape
                        if before.shape[1]<=anchor:
                            assert bit_equal(before,changed);witness['cached_checks']+=1;return
                        expected=before[:,anchor,channels]*gain if operation=='gain'else torch.tensor([donor[c['flat']]for c in cs],device=before.device,dtype=before.dtype).expand(before.shape[0],-1)
                        assert bit_equal(changed[:,anchor,channels],expected)
                        outside=changed.clone();outside[:,anchor,channels]=before[:,anchor,channels]
                        assert bit_equal(outside,before)
                        witness['prefill_checks']+=1
                        witness['prefills'].append({'layer':layer,'before_target_sha256':digest(before[:,anchor,channels]),'after_target_sha256':digest(changed[:,anchor,channels]),'expected_target_sha256':digest(expected),'other_values_bit_exact':True})
                    after_handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(after))
                yield {'edit':edit,'scope':witness}
            finally:
                for handle in after_handles:handle.remove()
    finally:
        for handle in handles:handle.remove()
        pending.clear()
