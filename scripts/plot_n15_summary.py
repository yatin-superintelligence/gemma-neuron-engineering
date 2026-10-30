"""Show the complete N15 coding grid and every frozen layer-subset gate."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'runs/n15/version1/descriptive_results.json';private=source.exists()
    if not private:source=ROOT/'results/n15_layer_pair_summary.json'
    data=json.loads(source.read_text());assert data['predictions']==920 and data['all_raw_token_grades_and_guards_pass']
    conditions=data['conditions'];gates=data['estimands']['site_gates'];families=['filter','predicate','extremum','rounding','json','arithmetic']
    assert len(conditions)==23 and len(gates)==9 and not any(g['eligible']for g in gates)
    raw=np.array([[c['families'][f]['correct']for f in families]for c in conditions]);denom=np.array([8,8,8,8,4,4])
    control=np.array([[g['opposite_alternative_correct'],g['same_rule_preserved']]for g in gates])
    labels=['Original (carried)','Identity (carried)','Full block: opposite (carried)','Full block: same rule (carried)','Full block: recipient (carried)']
    for name in [c['condition']for c in conditions[5:]]:
        label=name.replace('_samepolicy',': same rule').replace('_opposite',': opposite')
        for i in range(5,10):label=label.replace(f'L{i:02}',f'Layer {i}')
        for i in range(5,9):label=label.replace(f'P{i:02}_{i+1:02}',f'Layers {i} to {i+1}')
        labels.append(label)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,12));grid=fig.add_gridspec(1,2,left=.26,right=.96,top=.81,bottom=.19,wspace=.46,width_ratios=[2.5,1])
    ax=fig.add_subplot(grid[0,0]);ax.imshow(raw/denom,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_yticks(range(23),labels,fontsize=9);ax.set_xticks(range(6),['Filtering','Predicate','Extremum','Rounding','JSON','Arithmetic'],rotation=28,ha='right')
    ax.set_title('A. Intended-rule correctness: all 23 conditions',loc='left',weight='bold',pad=14,fontsize=11)
    for i in range(23):
        for j in range(6):ax.text(j,i,f'{raw[i,j]}/{denom[j]}',ha='center',va='center',fontsize=8,weight='bold',color='white'if raw[i,j]/denom[j]>.6 else'#172c40')
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,23,1),minor=True);ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
    ax.axhline(4.5,color='#7b3b13',lw=1.8)
    ax=fig.add_subplot(grid[0,1]);ax.imshow(control,vmin=0,vmax=16,cmap='Blues',aspect='auto')
    ax.set_yticks(range(9),['Layer '+str(g['layers'][0])if len(g['layers'])==1 else f"Layers {g['layers'][0]} to {g['layers'][-1]}"for g in gates],fontsize=9)
    ax.set_xticks([0,1],['Opposite-rule\ntransfer','Same-rule\npreservation']);ax.set_title('B. Role controls / 16',loc='left',weight='bold',pad=14,fontsize=11)
    for i in range(9):
        for j in range(2):ax.text(j,i,str(control[i,j]),ha='center',va='center',weight='bold',fontsize=12,color='white'if control[i,j]>8 else'#172c40')
    fig.text(.035,.95,'N15 | No single layer or adjacent pair passes the fixed gate',fontsize=18,weight='bold',color='#18333c')
    fig.text(.035,.906,'The full five-layer parent site transfers 14/16 programs. Tested smaller sites transfer at most 6/16.',fontsize=12)
    fig.text(.035,.866,'Every same-rule candidate preserves 16/16. All conditions retain 8/8 JSON and arithmetic controls.',fontsize=11,color='#52636d')
    fig.text(.035,.125,'Gate unchanged: transfer at least 8; preservation at least 14; both families, labels, mappings and directions; all retention correct.',fontsize=9)
    fig.text(.035,.089,'920 audited records: 720 new and 200 exact carried parents. 360 prefill self, 108 cached self and 46 restoration checks pass.',fontsize=9)
    fig.text(.035,.053,'Zero backward passes and optimizer updates. Reused development. No individual neuron meaning, formal synergy or confirmed coding improvement.',fontsize=9)
    out=ROOT/('research/figures'if private else'docs/figures');out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in ['png','svg']:
        p=out/f'n15_summary.{suffix}';fig.savefig(p,dpi=150,facecolor='white')
        if suffix=='svg':p.write_text('\n'.join(line.rstrip()for line in p.read_text().splitlines())+'\n')
        paths.append(p)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all23conditions_and6families_represented':True,'all9site_gates_represented':True,'new_model_forwards':0,'optimizer_steps':0,'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths},'visual_QA':'pending'}
    if private:(ROOT/'runs/n15/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
