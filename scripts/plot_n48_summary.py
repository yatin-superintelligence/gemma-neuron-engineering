"""Render the complete audited N48 aggregate once; no model or scorer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    folder=ROOT/'runs/n48/reporting';freeze=read(folder/'source_freeze.json')
    for n,h in freeze['sources'].items():assert sha(ROOT/n)==h,n
    aggregate=ROOT/'runs/n48/version1/public_aggregate.json';audit=ROOT/'runs/n48/version1/result_receipt.json'
    receipt=read(audit);data=read(aggregate)
    assert receipt['status']=='complete_pass_all_frozen_guards' and receipt['public_aggregate_sha256']==sha(aggregate)
    assert receipt['all24_contexts_120_condition_records'] is True and data['condition_records']==120
    modes=['original','remove_negative','random_negative_count','remove_positive','random_positive_count']
    labels=['Original','Negative','Random neg.','Positive','Random pos.'];colors=['#244b68','#077f77','#67aaa2','#b96d30','#d5ae80']
    target=ROOT/'research/figures/n48/n48_summary.png';target.parent.mkdir(parents=True,exist_ok=False)
    assert not (folder/'actual_input_binding.json').exists()
    (folder/'actual_input_binding.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'aggregate_sha256':sha(aggregate),'audit_sha256':sha(audit),'reporting_freeze_sha256':sha(folder/'source_freeze.json'),'reporting_after_complete_audit':True,'new_model_tokenizer_scorer_calls':0},indent=2)+'\n')
    groups=data['conditions'];x=np.arange(5)
    fig,axes=plt.subplots(3,2,figsize=(18,16));fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.08,hspace=.43,wspace=.40)
    fig.suptitle('N48 | Complete answer-decision panel; both signed primary tests fail',fontsize=19,fontweight='bold')
    for ax,role,total,title in [(axes[0,0],'coding',16,'A. Code comprehension'),(axes[0,1],'retention',8,'B. Retention')]:
        values=[groups[m][role+'_correct'] for m in modes];assert all(groups[m][role+'_cases']==total for m in modes)
        ax.bar(x,values,color=colors);ax.set_xticks(x,labels,rotation=20);ax.set_ylim(0,total+1.5);ax.set_ylabel('Correct first answers / '+str(total));ax.set_title(title,loc='left',fontweight='bold')
        for i,v in enumerate(values):ax.text(i,v+.2,str(v),ha='center')
        ax.spines[['top','right']].set_visible(False)
    ax=axes[1,0];gains=[groups[m]['coding_gains'] for m in modes[1:]];losses=[groups[m]['coding_losses'] for m in modes[1:]]
    ax.bar(np.arange(4)-.17,gains,width=.34,label='Gains',color='#077f77');ax.bar(np.arange(4)+.17,losses,width=.34,label='Losses',color='#b66a40')
    ax.set_xticks(np.arange(4),labels[1:],rotation=20);ax.set_ylim(0,4);ax.set_ylabel('Coding cases vs original');ax.set_title('C. All repairs and regressions',loc='left',fontweight='bold');ax.legend(frameon=False)
    ax=axes[1,1];families=sorted(groups['original']['families'],key=lambda f:(groups['original']['families'][f]['role'],f))
    matrix=np.array([[groups[m]['families'][f]['correct'] for m in modes] for f in families]);ax.imshow(matrix,vmin=0,vmax=2,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(np.arange(12),[f.replace('_',' ') for f in families]);ax.set_xticks(x,labels,rotation=20)
    for i in range(12):
        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]==2 else '#172c40',fontsize=10)
    ax.set_title('D. Every family, correct / 2 answer orders',loc='left',fontweight='bold')
    ax=axes[2,0]
    ax.plot(x,[groups[m]['coding_correct_margin_case_mean'] for m in modes],marker='o',label='Coding / 16',color='#244b68')
    ax.plot(x,[groups[m]['retention_correct_margin_case_mean'] for m in modes],marker='s',label='Retention / 8',color='#077f77')
    ax.axhline(0,color='#b2bdc5',linewidth=1);ax.set_xticks(x,labels,rotation=20);ax.set_ylabel('Mean correct-minus-foil logit');ax.set_title('E. Option margins, not full-program utility',loc='left',fontweight='bold');ax.legend(frameon=False)
    ax=axes[2,1];ax.axis('off')
    primary=data['signed_policy_primary'];assert all(primary[m]['passed'] is False for m in ['remove_negative','remove_positive'])
    lines=['F. Preserved primary failures','', 'Negative: 10/16; matched random: 10/16.', 'Three gains and one regression in each.', 'Only one program gains both answer orders.', '', 'Positive: 8/16; matched random: 9/16.', 'Two gains, two regressions; zero paired gains.', '', 'All 120 choices are valid A/B. All retention: 8/8.', 'All gains repair valid but wrong original choices.', 'Full clause tables remain in the audited JSON.']
    ax.text(0,1,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=12,linespacing=1.5)
    fig.text(.12,.018,'24 host-authored development contexts; paired answer orders are dependent. Current final-MLP sign masks match random cardinality, not energy.\nFirst full-vocabulary answer decisions only. Zero emitted tokens and optimizer updates. No selective signed advantage, individual meaning or useful full-program gain.',fontsize=11)
    fig.savefig(target,dpi=150);plt.close(fig)
    (folder/'actual_render_receipt.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'figure_path':str(target.relative_to(ROOT)),'figure_sha256':sha(target),'aggregate_sha256':sha(aggregate),'actual_render_invocations':1,'all6panels_viewed':False,'new_model_tokenizer_scorer_calls':0},indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
