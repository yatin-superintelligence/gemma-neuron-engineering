"""Strict profile matching; identical selection function from the frozen source."""
import hashlib,random
import numpy as np

def select(data,widths,count=64,seeds=(34201,34202),minimum_pool=16,max_excluded=512):
    offsets=np.cumsum([0]+list(widths));total=int(offsets[-1]);names=['RMS','delta_RMS','down_norm','min_score'];v={k:np.asarray(data[k],dtype=np.float64)for k in names}
    assert all(x.shape==(total,)and np.isfinite(x).all()for x in v.values())
    eligible=(v['min_score']>0)&(v['RMS']>1e-12)&(v['delta_RMS']>1e-12)&(v['down_norm']>1e-12)
    order=sorted(np.flatnonzero(eligible).tolist(),key=lambda k:(-float(v['min_score'][k]),k));chosen=[];excluded=[];final=None
    def coordinate(flat):
        layer=int(np.searchsorted(offsets,flat,side='right')-1)
        return{'flat':int(flat),'layer':layer,'channel':int(flat-offsets[layer]),'score':float(v['min_score'][flat]),'RMS':float(v['RMS'][flat]),'delta_RMS':float(v['delta_RMS'][flat]),'down_norm':float(v['down_norm'][flat])}
    def controls(proposal):
        result={}
        for seed in seeds:
            rng=random.Random(seed);used=set();group=[]
            for flat in proposal:
                layer=coordinate(flat)['layer'];lo,hi=map(int,offsets[layer:layer+2]);indices=np.arange(lo,hi)
                mask=(v['RMS'][lo:hi]>=.9*v['RMS'][flat])&(v['RMS'][lo:hi]<=1.1*v['RMS'][flat])&(v['delta_RMS'][lo:hi]>=.9*v['delta_RMS'][flat])&(v['delta_RMS'][lo:hi]<=1.1*v['delta_RMS'][flat])&(v['down_norm'][lo:hi]>=.8*v['down_norm'][flat])&(v['down_norm'][lo:hi]<=1.25*v['down_norm'][flat])
                blocked=[i-lo for i in set(proposal)|used if lo<=i<hi];mask[blocked]=False;pool=indices[mask].tolist()
                if len(pool)<minimum_pool:return None,{'seed':seed,'candidate_matched_to':flat,'pool_size':len(pool),'minimum_pool':minimum_pool,'proposal_size':len(proposal)}
                other=rng.choice(pool);used.add(other);group.append({**coordinate(other),'matched_to':flat,'pool_size':len(pool),'pool_sha256':hashlib.sha256(np.asarray(pool,dtype='<i8').tobytes()).hexdigest(),'RMS_ratio':float(v['RMS'][other]/v['RMS'][flat]),'delta_RMS_ratio':float(v['delta_RMS'][other]/v['delta_RMS'][flat]),'down_norm_ratio':float(v['down_norm'][other]/v['down_norm'][flat])})
            result[f'random{seed}_{count}']=group
        return result,None
    for flat in order:
        matched,reason=controls(chosen+[flat])
        if matched is None:
            excluded.append({'rejected':coordinate(flat),'reason':reason})
            if len(excluded)>max_excluded:break
        else:chosen.append(flat);final=matched
        if len(chosen)==count:break
    base={'positive_candidates':len(order),'ordered_positive_sha256':hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),'excluded_candidates':excluded,'accepted':len(chosen),'required':count,'optimizer_steps':0,'selection_uses_new_coding_outcomes':False}
    if len(chosen)!=count:return{**base,'status':'planned_host_matching_stop','groups':{}}
    groups={f'selected{count}':[coordinate(i)for i in chosen],**final}
    selected=set(chosen)
    for name,group in groups.items():
        assert len(group)==len({c['flat']for c in group})==count
        if name.startswith('random'):
            assert not selected&{c['flat']for c in group}
            for a,b in zip(groups[f'selected{count}'],group):assert a['layer']==b['layer']and b['matched_to']==a['flat']
    nested=min(16,count)
    for name,group in list(groups.items()):groups[name.rsplit('_',1)[0]+'_'+str(nested)if name.startswith('random')else'selected'+str(nested)]=group[:nested]
    return{**base,'status':'matched','groups':groups,'layer_histogram':{str(i):sum(c['layer']==i for c in groups[f'selected{count}'])for i in range(len(widths))},'cross_random_overlap':len({c['flat']for c in final[f'random{seeds[0]}_{count}']}&{c['flat']for c in final[f'random{seeds[1]}_{count}']})}
