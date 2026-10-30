"""The complete stage/site grid,with layout fixed before output access."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/'runs/n22/version1/result_receipt.json').read_text()); assert result['all273records_and_mandatory_guards_pass']
    data=json.loads((ROOT/'runs/n22/version1/descriptive_results.json').read_text()); cells=data['stage_by_site_cells']; assert len(cells)==4
    labels=['Last input\nLayers 5, 6, 9','Last input\nAll 35 layers','Current decision\nLayers 5, 6, 9','Current decision\nAll 35 layers']
    figure,axes=plt.subplots(2,1,figsize=(10,8.5),gridspec_kw={'height_ratios':[1.2,2]})
    axis=axes[0]; xs=list(range(4)); axis.bar([x-.18 for x in xs],[r['opposite_correct'] for r in cells],width=.36,label='Opposite-policy transfer',color='#007e7a'); axis.bar([x+.18 for x in xs],[r['same_policy_preserved'] for r in cells],width=.36,label='Same-policy preservation',color='#8a9b9c')
    axis.set(ylim=(0,13.9),yticks=[0,4,8,13],ylabel='Correct programs /13',xticks=xs,xticklabels=labels); axis.legend(frameon=False,ncol=2,fontsize=9,loc='upper left',bbox_to_anchor=(0,1.15))
    for index,cell in enumerate(cells):
        for offset,key in [(-.18,'opposite_correct'),(.18,'same_policy_preserved')]:
            axis.text(index+offset,cell[key]+.15,str(cell[key]),ha='center',fontsize=9)
    table=axes[1]; table.axis('off'); names={c['condition']:c for c in data['conditions']}
    order=['base','end_opposite','end_samepolicy','end_recipient','full_opposite','full_samepolicy','full_self','all35_end_opposite','all35_end_samepolicy','three_current_opposite','three_current_samepolicy','all35_end_self','three_current_self']
    rows=[]
    for name in order:
        row=names[name]; kinds=row['record_kinds']; kind='Parent' if kinds['exact_parent_carry'] else 'Self reference' if kinds['guarded_self_reference'] else '13 new /8 N/A'
        rows.append([name.replace('_',' '),kind,str(row['role_correct']),str(row['opposite_correct']),str(row['coding_gains']),str(row['coding_losses'])])
    artist=table.table(cellText=rows,colLabels=['All 13 fixed policies','Record type','Own/13','Opposite/13','Gains','Losses'],loc='center',cellLoc='center',colWidths=[.36,.22,.12,.12,.09,.09]); artist.auto_set_font_size(False); artist.set_fontsize(8); artist.scale(1,1.5)
    figure.suptitle('N22: intervention stage and native MLP coverage',fontsize=14,fontweight='bold')
    figure.text(.5,.025,'Same 13 inherited development cases. Stage includes generated prefix and cache state.\nForeign-donor retention is N/A, not successful intervention retention. All original grades stay fixed.\nDonor transfer is not original-model improvement, individual neuron meaning or formal interaction.',ha='center',fontsize=9)
    figure.subplots_adjust(left=.09,right=.98,top=.91,bottom=.12,hspace=.31)
    folder=ROOT/'research/figures'; figure.savefig(folder/'n22_summary.png',dpi=160); figure.savefig(folder/'n22_summary.svg'); plt.close(figure)


if __name__=='__main__':
    main()
