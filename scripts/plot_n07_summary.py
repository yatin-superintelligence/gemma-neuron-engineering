"""Every individual layer under both rule and reversed-label controls."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    private=ROOT/'runs/n07/version1/result_receipt.json'
    source=private if private.exists()else ROOT/'results/n07_layer_summary.json'
    data=json.loads(source.read_text());assert data['predictions']==600 and data['all_raw_token_grades_and_guards_pass']
    scores={r['condition']:r for r in data['conditions']}
    assert len(scores)==30 and data['eligible_layers']==[]
    opp=np.array([scores[f'opposite_L{i:02}']['opposite_correct']for i in range(14)])
    same=np.array([scores[f'samepolicy_L{i:02}']['opposite_correct']for i in range(14)])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(12.5,6.8));fig.subplots_adjust(left=.08,right=.97,top=.75,bottom=.31)
    x=np.arange(14);width=.37
    ax.bar(x-width/2,opp,width,color='#d37a39',label='Opposite-rule donor')
    ax.bar(x+width/2,same,width,color='#576c98',label='Same-rule donor, reversed label mapping')
    for i,(a,b)in enumerate(zip(opp,same)):
        if a:ax.text(i-width/2,a+.3,str(a),ha='center',fontsize=10,weight='bold',color='#a75115')
        if b:ax.text(i+width/2,b+.3,str(b),ha='center',fontsize=10,weight='bold',color='#364d7b')
    ax.set_xticks(x);ax.set_xlabel('Native MLP layer (one whole-layer swap at the selector)')
    ax.set_ylim(0,18);ax.set_yticks([0,4,8,12,16]);ax.set_ylabel('Programs switched to the opposite rule / 16')
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(-.01,1.20),frameon=False,ncol=2,fontsize=10)
    fig.text(.04,.945,'N07 | Layer 0 fails the rule-specificity control',fontsize=19,weight='bold',color='#18333c')
    fig.text(.04,.885,'All 14 individual layers and both donors. No layer meets the predefined eligibility criterion.',fontsize=11,color='#52636d')
    fig.text(.04,.205,'Original and identity: 16/16 correct. Every donor condition: 4/4 JSON controls correct, with no token-cap cases.',fontsize=10,color='#364b56')
    fig.text(.04,.158,'Layer 0 switches all programs under both donors, including donors that preserve the intended rule while changing the label.',fontsize=10,color='#364b56')
    fig.text(.04,.111,'The two switches at layers 4 and 9 each occur in one direction/mapping stratum. Nuisance names are repeated instances.',fontsize=10,color='#364b56')
    fig.text(.04,.064,'600 audited records, including 20 carried originals. Zero training. No abstract-rule neuron or coding improvement established.',fontsize=10,color='#364b56')
    out=ROOT/('research/figures'if private.exists()else'docs/figures');out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in ['png','svg']:
        p=out/f'n07_summary.{suffix}';fig.savefig(p,dpi=180,facecolor='white');paths.append(p)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'figure_prepared_visual_QA_pending',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all30conditions_represented':True,'new_model_forwards':0,'optimizer_steps':0,
        'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths}}
    if private.exists():(ROOT/'runs/n07/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
