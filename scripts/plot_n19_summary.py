"""Plot the complete N19 condition grid from audited aggregate counts."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    source=ROOT/'runs/n19/version2/descriptive_results.json';source=source if source.exists()else ROOT/'results/n19_current_decision_summary.json';r=json.loads(source.read_text());rows=r['estimands']['conditions'];names=[x['condition']for x in rows];n=len(rows);y=np.arange(n)
    correct=[sum(x['families'][f]['correct']for f in ['filter','predicate'])for x in rows];opposite=[sum(x['families'][f]['opposite_correct']for f in ['filter','predicate'])for x in rows]
    labels=[name+(' [carry]'if name=='base'else' [reference]'if name=='full_self'or name.endswith('_rescue')else'')for name in names]
    fig,axes=plt.subplots(1,3,figsize=(12.7,9.2),sharey=True,gridspec_kw={'width_ratios':[1,1,.9]});fig.subplots_adjust(left=.29,right=.98,top=.87,bottom=.15,wspace=.22)
    for ax,values,title,color in zip(axes[:2],[correct,opposite],['Original policy correct / 13','Opposite policy correct / 13'],['#246b62','#c16d1b']):
        ax.barh(y,values,color=color,height=.65)
        for i,v in enumerate(values):ax.text(v+.12,i,str(v),va='center',fontsize=9)
        ax.set_xlim(0,14.2);ax.set_xticks([0,4,8,13]);ax.set_title(title,fontsize=11,pad=14);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    axes[0].set_yticks(y,labels,fontsize=9);axes[0].invert_yaxis()
    ax=axes[2];ax.set_title('Retention correct / 8',fontsize=11,pad=14)
    for i,row in enumerate(rows):
        if row['record_kinds']['nonapplicable_reference']:
            ax.text(.15,i,'Not applicable',va='center',fontsize=8.5,color='#626b73');continue
        value=row['families']['json']['correct']+row['families']['arithmetic']['correct'];reference=bool(row['record_kinds']['carry']or row['record_kinds']['guarded_reference'])
        ax.barh(i,value,color='#a9b8bf'if reference else'#246b62',height=.65,hatch='///'if reference else None)
        ax.text(value+.1,i,str(value),va='center',fontsize=9)
    ax.set_xlim(0,9.7);ax.set_xticks([0,4,8]);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    for ax in axes:
        for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
        ax.tick_params(axis='y',length=0)
    fig.suptitle('Current-state donor control succeeds; three fixed neurons remain null',fontsize=15,fontweight='bold',x=.5,y=.965)
    fig.text(.29,.915,'N19: all 19 conditions retained. Target L9C4632 and controls L9C850 / L9C6085.',fontsize=10,color='#354550')
    fig.text(.02,.095,'399 records: 230 new edited tails, 21 parent carries and 148 references (64 donor cells not applicable).',fontsize=10)
    fig.text(.02,.067,'Hatching marks carried or guarded reference outcomes. Non-applicable donor cells are not retention tests.',fontsize=9.5)
    fig.text(.02,.038,'Reused development cases; controls are not activity matched. No individual neuron role or coding improvement established.',fontsize=9.5,color='#354550')
    out=ROOT/'output/figures/n19_summary.png';out.parent.mkdir(exist_ok=True,parents=True);fig.savefig(out,dpi=170,facecolor='white');plt.close(fig)
    target=ROOT/'research/figures/n19_summary.png';target.write_bytes(out.read_bytes());receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'all19conditions':True,'nonapplicable_cells_not_scored_as_retention':True,'visual_QA':'pending'};(ROOT/'runs/n19/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
