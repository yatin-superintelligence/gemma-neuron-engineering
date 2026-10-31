"""Plot every audited conditional-completion outcome from N33."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=root/'runs/n33/phase_A/version1/public_aggregate.json')
    parser.add_argument('--output',type=Path,default=root/'research/figures/n33_summary')
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    assert data['interventions']==0 and len(data['all8blocks'])==8
    plt.rcParams.update({'font.size':11,'svg.hashsalt':'n33-all-conditional-baseline-cases'})
    fig,axes=plt.subplots(3,1,figsize=(12,10),height_ratios=[1,1.3,.85],layout='constrained')
    colors=['#087f8c','#cf8665'];families=data['family_outcomes'];a=axes[0]
    correct=np.array([r['correct']for r in families]);wrong=np.array([r['cases']-r['correct']for r in families])
    a.bar(range(3),correct,color=colors[0],label='Correct under frozen contract');a.bar(range(3),wrong,bottom=correct,color=colors[1],label='Executable wrong answer')
    for i,(x,y)in enumerate(zip(correct,wrong)):
        for v,b in [(x,0),(y,x)]:
            if v:a.text(i,b+v/2,str(v),ha='center',va='center',color='white',weight='bold')
    a.set_xticks(range(3),['Predicate /16','Filter /16','Retention /8']);a.set_ylim(0,18);a.set_ylabel('Assembled conditional programs');a.legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1));a.set_title('All 40 supplied-scaffold completions',loc='left',weight='bold')
    a=axes[1];blocks=data['all8blocks'];correct=np.array([b['correct']for b in blocks]);wrong=4-correct
    a.barh(range(8),correct,color=colors[0]);a.barh(range(8),wrong,left=correct,color=colors[1])
    for i,(x,y)in enumerate(zip(correct,wrong)):
        for v,b in [(x,0),(y,x)]:
            if v:a.text(b+v/2,i,str(v),ha='center',va='center',color='white',weight='bold')
    a.set_yticks(range(8),[f"{b['family'].title()}, wording {b['style']}, name {b['name_index']}"for b in blocks]);a.invert_yaxis();a.set_xticks(range(5));a.set_xlim(0,4);a.set_title('All eight dependent four-case blocks; none selected for later testing',loc='left',weight='bold')
    a=axes[2];a.axis('off');receipt=data['phase_A']
    a.text(0,.98,'Predefined gate: STOP\n'
        'Role 28/32 and retention 3/8; required 32/32 and 8/8. One output reaches the 48-token cap.\n'
        'Required operator-only first token: 28/32. All fixed input scaffolds and token alternatives pass.\n'
        'Four predicate errors request the larger relation with reversed operands; all original grades remain fixed.\n'
        '40 new conditional tails; 344 forwards, 4512 input-token exposures and 344 generated tokens.\n'
        f"Worker {receipt['worker_seconds']:.3f} s; notebook {receipt['notebook_seconds']:.3f} s, not charged quota.\n"
        'All audit, cache, parameter, buffer, configuration, hook and counter guards pass; zero backwards or updates.\n'
        'No neuron intervention or B/C phase runs. This is not unrestricted coding or a group-effect null.',va='top',linespacing=1.6,fontsize=10.2)
    fig.suptitle('N33 | Supplied-scaffold baseline stops before neuron edits',weight='bold',fontsize=16)
    target=args.output;target.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg']:fig.savefig(target.with_suffix('.'+suffix),dpi=140,metadata={'Date':None}if suffix=='svg'else{})
    plt.close(fig)


if __name__=='__main__':main()
