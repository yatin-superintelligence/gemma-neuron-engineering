"""Compose full native MLP donors and explicit sparse native-channel edits."""
from contextlib import contextmanager,ExitStack
from interpretability.mlp_composition import compose_mlp_donors
from interpretability.anchor_intervention import anchor_intervention


@contextmanager
def native_group_policy(model,position,condition,donors,coordinates=None,block=range(5,10)):
    """Apply a declared policy using caller-supplied unmodified donor states.

    Donor keys are recipient, samepolicy, opposite and background_self. Each
    state has the capture_selector MLP mapping. Raw donor values are not bundled
    with a model recipe. A conditional cut in a full-donor context measures that
    engineered context, rather than necessity in the unmodified model.
    """
    assignments=[]
    if condition['background']:assignments.append({'layers':[0],'donor':donors['samepolicy']})
    if condition['full_donor']:assignments.append({'layers':list(block),'donor':donors[condition['full_donor']]})
    with ExitStack()as stack:
        receipt={'full':stack.enter_context(compose_mlp_donors(model,position,assignments)),'sparse':[],'optimizer_steps':0}
        operation=condition['operation']
        if operation!='none':
            if not coordinates:raise ValueError('Native coordinates required')
            def enter(op,kind=None,gain=1):
                values={c['flat']:float(donors[kind]['MLP'][c['layer']][0,c['channel']])for c in coordinates}if kind else None
                receipt['sparse'].append(stack.enter_context(anchor_intervention(model,coordinates,position,operation=op,gain=gain,donor=values)))
            if operation=='gain':enter('gain',gain=condition['gain'])
            elif operation in['background_self','samepolicy','opposite','recipient']:enter('swap',operation)
            elif operation=='opposite_then_same':enter('swap','opposite');enter('swap','samepolicy')
            else:raise ValueError('Unknown declared operation')
        yield receipt
