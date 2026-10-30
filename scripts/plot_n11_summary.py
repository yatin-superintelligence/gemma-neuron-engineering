"""Complete group and individual native-neuron development outcomes."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'runs/n11/version1/public_summary.json';private=source.exists()
    if not private:source=ROOT/'results/n11_neuron_summary.json'
    data=json.loads(source.read_text());assert data['predictions']==1152 and not data['primary_criterion_passed']
    scores={r['condition']:r['filter_correct']for r in data['conditions']};single={r['condition']:r['correct']for r in data['individual_conditions']}
    groups=['selected','random9101','random9102'];arms=['clean_zero','clean_double','corrupt_swap_same','corrupt_swap_opposite','cut_full']
    names=list(scores)[:10];matrix=np.array([[scores[f'{arm}_{group}_{size}']for group in groups]for size in[64,256]for arm in arms])
    individuals=np.array([[single[f'single_{arm}_{group}_{rank}']for group in groups for rank in range(8)]for arm in['corrupt_swap_same','cut_full']])
    assert matrix.shape==(10,3)and individuals.shape==(2,24)and len(scores)==40 and len(single)==48
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,11.5));grid=fig.add_gridspec(2,2,left=.21,right=.97,top=.84,bottom=.21,hspace=.9,wspace=.62,height_ratios=[3,1])
    ax=fig.add_subplot(grid[0,0]);y=np.arange(10)
    ax.barh(y,[scores[n]for n in names],color='#3f7b8f');ax.invert_yaxis();ax.set_xlim(0,18)
    ax.set_yticks(y,['Original','Layer 0 corruption','Clean full opposite','Clean full same','Corrupted full recipient','Corrupted full opposite','Corrupted full same','Clean sparse identity','Corrupted sparse identity','Full rescue / sparse restore'])
    for i,n in enumerate(names):ax.text(scores[n]+.25,i,str(scores[n]),va='center')
    ax.set_xticks([0,4,8,12,16]);ax.set_xlabel('Intended-rule code correct / 16');ax.set_title('A. Baselines and guard conditions',loc='left',weight='bold',pad=16)
    ax=fig.add_subplot(grid[0,1]);ax.imshow(matrix,vmin=0,vmax=16,cmap='Blues',aspect='auto')
    ax.set_xticks(range(3),['Selected','Random 9101','Random 9102'])
    labels=['Clean zero','Clean double','Same-rule sparse rescue','Opposite-rule sparse swap','Cut full rescue']
    ax.set_yticks(range(10),[f'{size}: {label}'for size in[64,256]for label in labels]);ax.set_title('B. Every sparse group condition',loc='left',weight='bold',pad=16)
    for i in range(10):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white'if matrix[i,j]>=10 else'#172c40',weight='bold')
    ax.axhline(4.5,color='#df9a4f',linewidth=2)
    ax=fig.add_subplot(grid[1,:]);ax.imshow(individuals,vmin=0,vmax=4,cmap='Blues',aspect='auto')
    ax.set_yticks([0,1],['Same-rule rescue','Cut full rescue']);ax.set_xticks(range(24),[str(i+1)for i in range(8)]*3)
    for i in range(2):
        for j in range(24):ax.text(j,i,str(individuals[i,j]),ha='center',va='center',color='white'if individuals[i,j]>=3 else'#172c40')
    for x in[7.5,15.5]:ax.axvline(x,color='white',linewidth=3)
    ax.set_xlabel('Selected ranks 1 to 8                 Random 9101 ranks 1 to 8                 Random 9102 ranks 1 to 8')
    ax.set_title('C. Every individual condition: correct / 4. Corresponding baselines: rescue 0/4, full rescue 3/4.',loc='left',weight='bold',pad=15)
    fig.text(.04,.957,'N11 | The predefined sparse-neuron criterion fails',fontsize=19,weight='bold',color='#18333c')
    fig.text(.04,.912,'Selected 256-unit group: 2 rescues and 4 conditional cut losses. Both matched random groups: 0 and 0.',fontsize=12)
    fig.text(.04,.88,'Required rescue advantage: at least 4, with both directions and mappings. Observed advantage: 2; coverage fails.',fontsize=11,color='#6e3b32')
    fig.text(.04,.135,'All 40 main conditions retain 4/4 JSON and 4/4 arithmetic. All 48 individual outcomes match their baselines and controls.',fontsize=11)
    fig.text(.04,.098,'1,152 audited records: 1,012 new, 140 carried. Reused development formulations with nuisance-name replications. No token caps.',fontsize=10)
    fig.text(.04,.061,'Cuts test dependence inside an engineered full-rescue context. They do not establish necessity in the original model.',fontsize=10)
    fig.text(.04,.025,'Zero optimizer updates. No original-model coding gain, individual semantic role or confirmed improvement is established.',fontsize=10)
    out=ROOT/('research/figures'if private else'docs/figures');out.mkdir(parents=True,exist_ok=True);paths=[]
    for suffix in['png','svg']:
        p=out/f'n11_summary.{suffix}';fig.savefig(p,dpi=150,facecolor='white')
        if suffix=='svg':p.write_text('\n'.join(s.rstrip()for s in p.read_text().splitlines())+'\n')
        paths.append(p)
    plt.close(fig)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all40main_and48individual_conditions_represented':True,'new_model_forwards':0,'optimizer_steps':0,'files':{str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths},'visual_QA':'pending'}
    if private:(ROOT/'runs/n11/version1/figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
