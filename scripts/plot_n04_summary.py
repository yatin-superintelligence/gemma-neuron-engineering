"""Render the complete reviewed N04 policy grid as a descriptive research figure."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,default=Path('results/n04_connection_summary.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/figures/n04_summary'))
    args=parser.parse_args();data=json.loads(args.summary.read_text())
    conditions=['original','off','kv25','kv100','key25','value_history25','value_history100','value_current25','value_permuted25']
    labels=['Original','Off','Keys + values, 0.25','Keys + values, 1.00','Keys only, 0.25',
        'Historical values, 0.25','Historical values, 1.00','Current values, 0.25','Permuted values, 0.25']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'gne-n04-fixed-grid'})
    fig,(left,right)=plt.subplots(1,2,figsize=(13,6.8),gridspec_kw={'width_ratios':[1,1.15]},sharey=True)
    y=np.arange(len(conditions))
    ranges=[data['path_logit_effect_ranges'][c]['selected_past']for c in conditions]
    lo=np.array([r['min']for r in ranges]);hi=np.array([r['max']for r in ranges])
    middle=(lo+hi)/2
    left.errorbar(middle,y,xerr=[middle-lo,hi-middle],fmt='o',capsize=4,color='#17665f',lw=2,markersize=5)
    left.axvline(0,color='#89979f',lw=1,zorder=0)
    left.set_yticks(y,labels);left.invert_yaxis();left.set_xlim(-.06,1.45)
    left.set_xlabel('Maximum absolute change in later logits')
    left.set_title('A. Past activity in eight selected native channels',loc='left',fontweight='bold',pad=18)
    left.text(.02,1.01,'Range over two fixed-input prompts, not a confidence interval',transform=left.transAxes,fontsize=8.5,color='#52646d')
    totals=np.zeros(len(conditions),dtype=int)
    for family,color,label in [('threshold','#17665f','Threshold (16)'),('division','#c48a2d','Signed division (8)'),('json','#8b9ca5','JSON (8)')]:
        counts=np.array([next(r['correct']for r in data['all_condition_family_results']if r['condition']==c and r['family']==family)for c in conditions])
        right.barh(y,counts,left=totals,color=color,height=.64,label=label);totals+=counts
    for row,total in enumerate(totals):right.text(total+.45,row,str(total),va='center',fontsize=9)
    right.set_xlim(0,32);right.xaxis.set_major_locator(MaxNLocator(integer=True))
    right.set_xlabel('Correct outputs, out of 32 development/retention prompts')
    right.set_title('B. Executable coding and retention',loc='left',fontweight='bold',pad=18)
    right.legend(loc='lower center',bbox_to_anchor=(.49,-.25),ncol=3,frameon=False,fontsize=8.5)
    for axis in [left,right]:
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='x',color='#dfe6e7',lw=.7);axis.set_axisbelow(True)
        axis.tick_params(axis='y',length=0)
    fig.suptitle('Copied memory opens a native-neuron path; coding benefit remains unconfirmed',
        x=.03,ha='left',fontsize=15,fontweight='bold',y=.98)
    fig.text(.03,.925,'Text-only unquantized BF16 Gemma 4 E2B. All nine fixed policies retained. Zero optimizer updates.',fontsize=10,color='#435761')
    fig.text(.03,.055,'The key-only result is two division gains on four reused function-name pairs (paired p = 0.5). No semantic neuron role is established.',fontsize=9)
    fig.text(.03,.025,'Cache precision uses the disclosed amended gate after a native-baseline failure. All exact off-mode and restoration controls pass.',fontsize=9,color='#52646d')
    fig.subplots_adjust(left=.225,right=.97,bottom=.27,top=.82,wspace=.26)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180,metadata={'Software':'Gemma Neuron Engineering'})
    fig.savefig(args.output.with_suffix('.svg'),metadata={'Date':None,'Creator':'Gemma Neuron Engineering'})
    plt.close(fig)


if __name__=='__main__':main()
