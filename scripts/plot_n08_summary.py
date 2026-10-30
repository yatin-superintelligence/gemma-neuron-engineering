"""All fixed native blocks, donor controls and the original selection rule."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    private=ROOT/'runs/n08/version1/result_receipt.json'
    source=private if private.exists()else ROOT/'results/n08_block_summary.json'
    data=json.loads(source.read_text());assert data['predictions']==660 and data['all_raw_token_grades_and_guards_pass']
    scores={r['condition']:r for r in data['conditions']};gates=data['block_gates']
    names=['clean_opposite_','clean_samepolicy_','corrupt0_recipient_','corrupt0_opposite_','corrupt0_samepolicy_']
    def name(prefix,block):return 'samepolicy_MLP'if prefix=='corrupt0_samepolicy_'and block=='B01_13'else prefix+block
    values=np.array([[scores[name(n,g['block_id'])]['target_correct']for n in names]for g in gates])
    assert values.shape==(6,5)and len(scores)==33 and data['selected_development_block']=='B05_09'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(12.5,7.4));fig.subplots_adjust(left=.13,right=.78,top=.72,bottom=.27)
    image=ax.imshow(values,vmin=0,vmax=16,cmap='Blues',aspect='auto')
    ax.set_xticks(range(5),['Clean:\nopposite rule','Clean:\nsame rule','Corrupted:\nrecipient','Corrupted:\nopposite rule','Corrupted:\nsame rule'])
    ax.set_yticks(range(6),['Layers '+','.join(str(i)for i in g['layers'])if len(g['layers'])<3 else f"Layers {min(g['layers'])} to {max(g['layers'])}"for g in gates])
    for i in range(6):
        for j in range(5):ax.text(j,i,str(values[i,j]),ha='center',va='center',weight='bold',color='white'if values[i,j]>=10 else'#172c40')
    for i,g in enumerate(gates):
        ax.text(4.7,i,'Selected'if g['block_id']=='B05_09'else'Eligible'if g['eligible']else'Fails gate',va='center',fontsize=10,
                color='#005d53'if g['eligible']else'#6b5960',weight='bold'if g['block_id']=='B05_09'else'normal')
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.04,.947,'N08 | Layers 5 to 9 pass the conditional rescue gate',fontsize=18,weight='bold',color='#18333c')
    fig.text(.04,.885,'Intended-rule code correct / 16. Every block and all five donor conditions are shown.',fontsize=11,color='#52636d')
    fig.text(.04,.828,'Original: 16/16. Fixed layer 0 corruption and its identity: 0/16. Every condition retains 4/4 JSON controls.',fontsize=10,color='#364b56')
    fig.text(.04,.177,'Eligibility: same-rule rescue at least 12; opposite-rule at most 4; recipient at least 12; clean same-rule at least 14.',fontsize=10,color='#364b56')
    fig.text(.04,.131,'The frozen rule selects the fewest layers, then earliest start. The highest rescue count does not determine selection.',fontsize=10,color='#364b56')
    fig.text(.04,.085,'660 audited records, including 60 carried references. Same development prompts with nuisance-name replications. Zero training.',fontsize=10,color='#364b56')
    fig.text(.04,.039,'A whole-block rescue of an induced error. Sparse neuron meaning, necessity and improvement of the original model remain unestablished.',fontsize=10,color='#364b56')
    out=ROOT/('research/figures'if private.exists()else'docs/figures');out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in ['png','svg']:
        p=out/f'n08_summary.{suffix}';fig.savefig(p,dpi=180,facecolor='white')
        if suffix=='svg':p.write_text('\n'.join(line.rstrip()for line in p.read_text().splitlines())+'\n')
        paths.append(p)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'figure_prepared_visual_QA_pending',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all33conditions_represented':True,'new_model_forwards':0,'optimizer_steps':0,
        'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths}}
    if private.exists():(ROOT/'runs/n08/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
