"""Plot every N21 arm and family after the immutable host audit passes."""
import datetime,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['base','identity','selected64_gain2','random34201_64_gain2','random34202_64_gain2']
LABELS=['Original','Identity','Selected64 ×2','Random A64 ×2','Random B64 ×2']


def main():
    receipt_path=ROOT/'runs/n21/version1/result_receipt.json';summary_path=ROOT/'runs/n21/version1/descriptive_results.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['status']=='complete_host_audited'and receipt['all160records_and_mandatory_guards_pass']
    summary=json.loads(summary_path.read_text());by={r['condition']:r for r in summary['conditions']};assert set(by)==set(CONDITIONS)
    families=sorted(by['base']['family_correct']);assert len(families)==8 and all(set(r['family_correct'])==set(families)for r in by.values())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(13,8),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1,1.15],width_ratios=[1,1.15])
    totals=fig.add_subplot(grid[0,0]);y=np.arange(5);correct=[by[c]['coding_correct']for c in CONDITIONS]
    totals.barh(y,correct,color=['#667780','#a8b9b6','#007c73','#9c6b40','#be9667']);totals.set_yticks(y,LABELS);totals.invert_yaxis();totals.set_xlim(0,25.5);totals.set_xticks([0,6,12,18,24]);totals.set_xlabel('Correct complete functions / 24');totals.set_title('All five fixed arms',loc='left',weight='bold')
    for i,c in enumerate(CONDITIONS):totals.text(correct[i]+.25,i,f"{correct[i]}/24",va='center',fontsize=9)
    pair=fig.add_subplot(grid[0,1]);comparators=['base','random34201_64_gain2','random34202_64_gain2'];names=['vs original','vs random A','vs random B'];x=np.arange(3)
    gains=[len(summary['contrasts'][c]['gains'])for c in comparators];losses=[len(summary['contrasts'][c]['losses'])for c in comparators]
    pair.bar(x-.17,gains,.34,color='#007c73',label='Gains');pair.bar(x+.17,[-v for v in losses],.34,color='#b75b46',label='Losses');pair.axhline(0,color='#334b52',linewidth=.8);pair.set_xticks(x,names);pair.set_ylim(-24,24);pair.set_yticks([-24,-12,0,12,24]);pair.set_ylabel('Paired functions / 24');pair.set_title('Selected64 ×2: gains and losses',loc='left',weight='bold');pair.legend(frameon=False,ncol=2)
    for i,c in enumerate(comparators):pair.text(i,21,f"net {summary['contrasts'][c]['net']:+d}",ha='center',fontsize=9)
    heat=fig.add_subplot(grid[1,:]);matrix=np.array([[by[c]['family_correct'][f]for f in families]for c in CONDITIONS]);heat.imshow(matrix,vmin=0,vmax=3,cmap='YlGnBu',aspect='auto')
    heat.set_yticks(range(5),LABELS);heat.set_xticks(range(8),[f.replace('_','\n')for f in families]);heat.set_title('All eight hand-authored families: correct / 3',loc='left',weight='bold')
    for i in range(5):
        for j in range(8):heat.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white'if matrix[i,j]>=2 else'#173a42')
    retention=', '.join(f"{label}: {by[c]['retention_correct']}/8"for label,c in zip(['original','identity','selected','random A','random B'],CONDITIONS))
    verdict='Passed'if summary['development_signal_gate_passed']else'Not passed'
    fig.suptitle(f'N21: fixed global native-column gain | Development screen: {verdict.lower()}',weight='bold',fontsize=15)
    fig.supxlabel('Retention: '+retention+'\nAll 24 coding cases retained. Eight families are dependence groups, not IID replications.\nA development signal requires fresh confirmation; it does not establish a semantic neuron role.',fontsize=9)
    out=ROOT/'research/figures';out.mkdir(exist_ok=True)
    files={}
    for ext in['png','svg']:
        p=out/f'n21_summary.{ext}';fig.savefig(p,dpi=160,metadata={'Creator':'Gemma Neuron Engineering'}if ext=='svg'else{'Software':'Gemma Neuron Engineering'});files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    plt.close(fig)
    value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'figures':files,'all5arms_and8families_retained':True,'all24coding_cases_retained':True,'raw_prompts_predictions_and_function_names_in_figure':False,'optimizer_steps':0,'visual_QA':'pending'}
    (ROOT/'runs/n21/version1/figure_receipt.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
