"""Current cached-token feasibility with actual random-weight Gemma."""
from pathlib import Path
import datetime,json,time
import torch
from tests import test_selector_site as fixture_module
from interpretability.decode_step import capture_decode_step,edit_decode_step
from interpretability.scoped_channel import bit_equal,digest


def main():
    start=time.monotonic();fixture=fixture_module.SelectorSiteTests();fixture.setUp();model=fixture.model;indices=list(range(6));prompt=fixture.ids[:,:3];other=prompt.clone();other[0,1]=9;token=torch.tensor([[5]])
    versions={n:p._version for n,p in model.named_parameters()};hook_counts={n:(len(m._forward_pre_hooks),len(m._forward_hooks))for n,m in model.named_modules()};meter={'calls':0}
    def count(module,args,kwargs):meter['calls']+=1
    handle=model.register_forward_pre_hook(count,with_kwargs=True)
    def fresh(source=prompt):
        with torch.no_grad():return model(source,use_cache=True).past_key_values
    donors=[];native_logits=[]
    for source in[prompt,other]:
        cache=fresh(source)
        with capture_decode_step(model,indices)as observation:
            with torch.no_grad():value=model(token,past_key_values=cache,use_cache=True)
        donors.append(observation['values']);native_logits.append(value.logits.detach().clone())
    cache=fresh()
    with capture_decode_step(model,indices,measurement_gradients=True)as measurement:
        with torch.enable_grad():value=model(token,past_key_values=cache,use_cache=True)
        assert bit_equal(value.logits,native_logits[0])
        metric=value.logits[0,-1,3]-value.logits[0,-1,7]
        gradients=torch.autograd.grad(metric,[measurement['graph'][i]for i in indices])
        assert all(torch.isfinite(g).all()and g.abs().sum()>0 for g in gradients)
    coordinates=[{'flat':i*100000+1,'layer':i,'channel':1}for i in indices]
    def source_values(source):return{c['flat']:float(donors[source][c['layer']][0,0,c['channel']])for c in coordinates}
    cases=[]
    for operation,gain,source in[('gain',0,0),('gain',1,0),('gain',2,0),('swap',1,0),('swap',1,1),('rescue',1,0)]:
        cache=fresh()
        with edit_decode_step(model,coordinates,operation=operation,gain=gain,donor=source_values(source))as receipt:
            with torch.no_grad():changed=model(token,past_key_values=cache,use_cache=True).logits
        exact=bit_equal(changed,native_logits[0]);assert exact==(operation=='rescue'or operation=='swap'and source==0 or operation=='gain'and gain==1)
        with torch.no_grad():restored=model(token,past_key_values=fresh(),use_cache=True).logits
        assert bit_equal(restored,native_logits[0]);cases.append({'operation':operation,'gain':gain,'source':source,'native_exact':exact,'receipt':receipt})
    for primitive in['capture','edit']:
        try:
            context=capture_decode_step(model,indices)if primitive=='capture'else edit_decode_step(model,coordinates)
            with context:
                with torch.no_grad():model(prompt,use_cache=True)
            raise AssertionError('Prefill was not rejected')
        except ValueError as error:assert 'cached token'in str(error)
        cache=fresh()
        try:
            context=capture_decode_step(model,indices)if primitive=='capture'else edit_decode_step(model,coordinates)
            with context:
                with torch.no_grad():value=model(token,past_key_values=cache,use_cache=True);model(token,past_key_values=value.past_key_values,use_cache=True)
            raise AssertionError('Repeated forward was not rejected')
        except ValueError as error:assert 'one forward'in str(error)
    with torch.no_grad():assert bit_equal(model(token,past_key_values=fresh(),use_cache=True).logits,native_logits[0])
    handle.remove();assert versions=={n:p._version for n,p in model.named_parameters()}and all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert hook_counts=={n:(len(m._forward_pre_hooks),len(m._forward_hooks))for n,m in model.named_modules()}
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'host_feasibility_passed','pretrained_model_forwards':0,'instrumented_tiny_model_forward_calls':meter['calls'],'fixture_setup_calls':2,'prescribed_interventions':len(cases),'gradient_measurement_backward_calls':1,'optimizer_steps':0,'all_parameter_gradients_none':True,'ordinary_instrumented_logits_bit_exact':True,'late_current_layer_gradients_nonzero':True,'prefill_and_second_call_rejected':True,'all_hooks_and_parameter_versions_restore':True,'gradient_hashes':[digest(g)for g in gradients],'seconds':time.monotonic()-start,'scope':'Tiny random-weight Gemma engineering proof,not a pretrained neuron role or new capability.'}
    folder=Path('runs/n19/host');folder.mkdir(parents=True,exist_ok=True);(folder/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
