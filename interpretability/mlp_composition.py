"""Compose disjoint native MLP donor assignments without changing weights."""
from contextlib import contextmanager,ExitStack
from interpretability.selector_site import selector_site


@contextmanager
def compose_mlp_donors(model,position,assignments):
    """Each assignment supplies native layer indices and actual captured states."""
    seen=set()
    for assignment in assignments:
        layers=list(assignment['layers'])
        if not layers or len(layers)!=len(set(layers))or seen.intersection(layers):
            raise ValueError('Nonempty disjoint native layer sets are required')
        if any(not isinstance(i,int)or not 0<=i<len(model.model.language_model.layers)for i in layers):
            raise ValueError('Invalid native layer index')
        seen.update(layers)
    receipts=[]
    with ExitStack()as stack:
        for assignment in assignments:
            receipt=stack.enter_context(selector_site(model,position,assignment['donor'],'MLP',mlp_layers=assignment['layers']))
            receipts.append({'layers':list(assignment['layers']),'receipt':receipt})
        yield {'parts':receipts,'optimizer_steps':0}
