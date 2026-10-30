"""All full-site conditions, coding families and prospective semantic gates."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'runs/n14/version1/descriptive_results.json';private=source.exists()
    if not private:source=ROOT/'results/n14_site_summary.json'
    data=json.loads(source.read_text());assert data['predictions']==400 and data['all_raw_token_grades_and_guards_pass']
    conditions=data['conditions'];families=['filter','predicate','extremum','rounding','json','arithmetic'];gates=data['estimands']['site_gates']
    assert len(conditions)==10 and len(gates)==3 and data['estimands']['selected_site']=='block5_9'
    raw=np.array([[c['families'][f]['correct']for f in families]for c in conditions]);denom=np.array([8,8,8,8,4,4])
    gate_values=np.array([[g['opposite_alternative_correct'],g['same_rule_preserved']]for g in gates])
    labels=['Original','Empty identity','Layer 0: opposite','Layer 0: same rule','Layers 5 to 9: opposite','Layers 5 to 9: same rule','Layers 5 to 9: recipient','Layers 0 to 13: opposite','Layers 0 to 13: same rule','Layers 0 to 13: recipient']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,8.6));grid=fig.add_gridspec(1,2,left=.23,right=.97,top=.76,bottom=.27,wspace=.45,width_ratios=[2.4,1])
    ax=fig.add_subplot(grid[0,0]);ax.imshow(raw/denom,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_yticks(range(10),labels);ax.set_xticks(range(6),['Filtering','Predicate','Extremum','Rounding','JSON','Arithmetic'],rotation=25,ha='right')
    ax.set_title('A. Intended-rule correctness: complete grid',loc='left',weight='bold',pad=16)
    for i in range(10):
        for j in range(6):ax.text(j,i,f'{raw[i,j]}/{denom[j]}',ha='center',va='center',weight='bold',color='white'if raw[i,j]/denom[j]>.6 else'#172c40',fontsize=10)
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',bottom=False,left=False)
    ax=fig.add_subplot(grid[0,1]);ax.imshow(gate_values,vmin=0,vmax=16,cmap='Blues',aspect='auto')
    ax.set_xticks([0,1],['Opposite-rule\ntransfer','Same-rule\npreservation']);ax.set_yticks([0,1,2],['Layer 0','Layers 5 to 9','Layers 0 to 13'])
    ax.set_title('B. Role controls / 16',loc='left',weight='bold',pad=16)
    for i in range(3):
        for j in range(2):ax.text(j,i,str(gate_values[i,j]),ha='center',va='center',weight='bold',color='white'if gate_values[i,j]>=10 else'#172c40',fontsize=14)
    for i,g in enumerate(gates):ax.text(1.55,i,'Pass'if g['eligible']else'Fail',va='center',weight='bold',color='#006b5e'if g['eligible']else'#8a4b3a',fontsize=10)
    fig.text(.035,.946,'N14 | Layers 5 to 9 pass the controlled rule-transfer test',fontsize=19,weight='bold',color='#18333c')
    fig.text(.035,.885,'Opposite-rule donors switch 14/16 programs. Reversed-label same-rule donors preserve all 16. Retention: 8/8.',fontsize=12)
    fig.text(.035,.827,'A full native site, not an identified individual neuron. Every declared condition and coding family is shown.',fontsize=11,color='#52636d')
    fig.text(.035,.185,'Gate: transfer at least 8; preservation at least 14; both domains, labels, mappings and directions; all retention correct.',fontsize=10)
    fig.text(.035,.139,'Layer 0 fails the semantic control. Layers 0 to 13 miss the preservation threshold. Selection uses the fixed eligibility rule.',fontsize=10)
    fig.text(.035,.093,'400 audited records, including 40 exact carried originals. 120 self checks, 120 generated identities and 20 restorations pass.',fontsize=10)
    fig.text(.035,.047,'All six nonrecipient donor policies have one incidental extremum gain. No original-model coding improvement is established.',fontsize=10)
    out=ROOT/('research/figures'if private else'docs/figures');out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in['png','svg']:
        p=out/f'n14_summary.{suffix}';fig.savefig(p,dpi=150,facecolor='white')
        if suffix=='svg':p.write_text('\n'.join(line.rstrip()for line in p.read_text().splitlines())+'\n')
        paths.append(p)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all10conditions_and6families_represented':True,'all3site_gates_represented':True,'new_model_forwards':0,'optimizer_steps':0,
        'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths},'visual_QA':'pending'}
    if private:(ROOT/'runs/n14/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
