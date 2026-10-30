"""Plot every N06 condition without inferring independent task counts."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    private=ROOT/'runs/n06/version1/result_receipt.json'
    source=private if private.exists()else ROOT/'results/n06_site_summary.json'
    data=json.loads(source.read_text())
    assert data['predictions']==200 and data['all_raw_token_grades_and_guards_pass']
    labels={'base':'Original','self_both':'Self swap: MLP + K/V',
        'opposite_K':'Opposite-rule donor: keys','opposite_V':'Opposite-rule donor: values',
        'opposite_KV':'Opposite-rule donor: K/V','opposite_MLP':'Opposite-rule donor: joint early MLP',
        'samepolicy_K':'Same-rule donor: keys','samepolicy_V':'Same-rule donor: values',
        'samepolicy_KV':'Same-rule donor: K/V','samepolicy_MLP':'Same-rule donor: joint early MLP'}
    rows=data['conditions'];assert list(labels)==[r['condition']for r in rows]
    current=np.array([r['target_correct']for r in rows]);opposite=np.array([r['opposite_correct']for r in rows])
    other=16-current-opposite;assert np.all(other>=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,(ax,ret)=plt.subplots(1,2,figsize=(13,7.8),gridspec_kw={'width_ratios':[5.6,1.1]})
    fig.subplots_adjust(left=.35,right=.96,top=.79,bottom=.24,wspace=.16)
    y=np.arange(10)
    ax.barh(y,current,color='#14736c',height=.64,label='Recipient rule')
    ax.barh(y,opposite,left=current,color='#d37a39',height=.64,label='Opposite rule')
    ax.barh(y,other,left=current+opposite,color='#a6b2bb',height=.64,label='Other / invalid')
    for i,(a,b,c)in enumerate(zip(current,opposite,other)):
        for left,value in [(0,a),(a,b),(a+b,c)]:
            if value:ax.text(left+value/2,i,str(value),ha='center',va='center',color='white',weight='bold')
    ax.set_yticks(y,[labels[r['condition']]for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0,16);ax.set_xticks([0,4,8,12,16]);ax.set_xlabel('Executable filter programs (16 prompts)')
    ax.set_axisbelow(True);ax.grid(axis='x',alpha=.16)
    ax.legend(loc='upper left',bbox_to_anchor=(-.01,1.16),ncol=3,frameon=False,fontsize=9)
    for i,r in enumerate(rows):ret.text(.5,i,f'{r["JSON_content_correct"]}/4',ha='center',va='center',color='#14736c',weight='bold')
    ret.set_ylim(ax.get_ylim());ret.set_xlim(0,1);ret.set_xticks([]);ret.set_yticks([])
    ret.set_title('JSON\nretention',fontsize=11,pad=10)
    for panel in [ax,ret]:
        for side in ['top','right','left']:panel.spines[side].set_visible(False)
        panel.tick_params(axis='y',length=0)
    ret.spines['bottom'].set_visible(False)
    fig.text(.035,.955,'N06 | Conditional policy control through native MLP activity',fontsize=18,weight='bold',color='#18333c')
    fig.text(.035,.905,'Text-only Gemma 4 E2B, unquantized BF16. All 10 fixed conditions. Zero training.',fontsize=11,color='#52636d')
    fig.text(.035,.155,'The joint MLP swap replaces 86,016 native channel values at one selector position across layers 0 through 13.',fontsize=10,color='#364b56')
    fig.text(.035,.119,'Same-rule donors reverse label mappings. Opposite-rule transfer changes the requested behavior; it is not a coding gain.',fontsize=10,color='#364b56')
    fig.text(.035,.083,'Sixteen prompts reuse four instruction/mapping strata with operand-name and policy pairs. They are not sixteen algorithms.',fontsize=10,color='#364b56')
    fig.text(.035,.047,'All 200 outputs and no-op/restoration guards passed audit. Individual neuron meanings and fresh transfer remain unresolved.',fontsize=10,color='#364b56')
    out=ROOT/('research/figures'if private.exists()else'docs/figures');out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ['png','svg']:
        path=out/f'n06_summary.{suffix}';fig.savefig(path,dpi=180,facecolor='white');paths.append(path)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'private_figure_prepared_visual_QA_pending',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all10conditions_in_protocol_order':True,'new_model_forwards':0,'optimizer_steps':0,
        'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths}}
    if private.exists():(ROOT/'runs/n06/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
