"""Complete saved native-component aggregates; no scorer or model imports."""
import math

DONORS=('paired_order_swap','unrelated_same_order')
CELLS=('10','01','11')
EFFECTS=('R','P','joint','interaction')
COUNTERS={'calls':312,'input_token_exposures':29510,'activation_assignments':144,'native_sourceguards':24,'native_restoration_forwards':144,'hybrid_component_constructions':144,'native_component_reconstruction_checks':312,'extra_FFN_evaluations':0,'extra_down_projection_evaluations':0,'new_emitted_tokens':0,'manual_parameter_assignments':0,'backward_calls':0,'optimizer_steps':0}


def require(test,message):
    if not test:raise ValueError(message)


def integer(value,wanted=None):
    require(type(value) is int and value>=0 and (wanted is None or value==wanted),'Strict actual integer count required')


def validate(data):
    require(data['experiment']=='N51' and data['status']=='complete_host_audited','Only complete audited N51 allowed')
    for key,wanted in [('contexts',24),('new_factor_records',144),('native_condition_carries',24),('new_literal_label_comparisons',144),('closed_scorer_calls',0)]:integer(data[key],wanted)
    require(set(data['actual_counters'])==set(COUNTERS),'Complete actual counter map required')
    for key,wanted in COUNTERS.items():integer(data['actual_counters'][key],wanted)
    require(set(data['all_donors'])==set(DONORS),'Both fixed donor conditions required')
    for donor,group in data['all_donors'].items():
        require(set(group)=={'roles','families','answer_orders'} and set(group['roles'])=={'coding','retention'},'Complete roles and supplementary groups required')
        require(len(group['families'])==12 and set(group['answer_orders'])=={'coding_0','coding_1','retention_0','retention_1'},'All families and answer orders required')
        selectors=[(group['roles']['coding'],16),(group['roles']['retention'],8)]+[(v,2) for v in group['families'].values()]+[(v,8 if k.startswith('coding') else 4) for k,v in group['answer_orders'].items()]
        for row,n in selectors:
            require(set(row)=={'signed_effects','absolute_effects','new_cell_scores'},'All effect and correctness summaries required')
            for kind in ['signed_effects','absolute_effects']:
                require(set(row[kind])==set(EFFECTS),'Every finite component contrast required')
                for stat in row[kind].values():
                    require(set(stat)=={'count','mean','min','max'},'Exact descriptive statistic fields required');integer(stat['count'],n)
                    require(all(type(stat[k]) in [int,float] and math.isfinite(stat[k]) for k in ['mean','min','max']),'Finite descriptive values required')
                    require(stat['min']<=stat['mean']<=stat['max'],'Mean outside actual descriptive range')
                    if kind=='absolute_effects':require(stat['min']>=0,'Absolute effects cannot be negative')
            require(set(row['new_cell_scores'])==set(CELLS),'Every new cell score required')
            for score in row['new_cell_scores'].values():
                require(set(score)=={'cases','correct','valid_options','matches_carried_recipient_choice','copies_native_donor_letter','complements_native_donor_letter','noncanonical_native_donor'},'Exact new-cell score fields required')
                for k,v in score.items():integer(v);require(v<=n,'Score count exceeds panel size')
                integer(score['cases'],n);integer(score['noncanonical_native_donor'],0)
                require(score['correct']<=score['valid_options'] and score['copies_native_donor_letter']+score['complements_native_donor_letter']==score['valid_options'],'Full-vocabulary validity/letter counts differ')
    original=data['carried_N48_aggregate']['conditions']['original']
    integer(original['coding_cases'],16);integer(original['retention_cases'],8)
    integer(original['coding_correct'],8);integer(original['retention_correct'],8)
    return data


def plot(data,path,*,synthetic=False):
    validate(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#087e77','#bd7538'];names=['Paired order swap','Unrelated same order'];keys=['R','P','joint'];labels=['Residual R','Per-layer P','Joint R + P'];x=np.arange(3)
    fig,axes=plt.subplots(3,2,figsize=(17,15));fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,hspace=.47,wspace=.23)
    fig.suptitle(('SYNTHETIC SOFTWARE LAYOUT | ' if synthetic else '')+'N51 native contribution cuts; recipient FFN fixed',fontsize=18,fontweight='bold')
    for row,kind in [(0,'absolute_effects'),(1,'signed_effects')]:
        for col,role in enumerate(['coding','retention']):
            ax=axes[row,col]
            for index,donor in enumerate(DONORS):
                values=[data['all_donors'][donor]['roles'][role][kind][key]['mean'] for key in keys]
                ax.bar(x+(index-.5)*.34,values,width=.34,color=colors[index],label=names[index])
            ax.set_xticks(x,labels);ax.axhline(0,color='#a8b5bb',linewidth=1);ax.spines[['top','right']].set_visible(False)
            ax.set_title(chr(65+row*2+col)+'. '+role.capitalize()+(' | absolute margin change' if row==0 else ' | signed margin change'),loc='left',fontweight='bold')
            ax.set_ylabel('Case mean, correct-minus-foil logits');ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2)
    for col,role in enumerate(['coding','retention']):
        ax=axes[2,col];n=16 if role=='coding' else 8
        for index,donor in enumerate(DONORS):
            values=[data['all_donors'][donor]['roles'][role]['new_cell_scores'][cell]['correct'] for cell in CELLS]
            bars=ax.bar(x+(index-.5)*.34,values,width=.34,color=colors[index],label=names[index])
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.12,str(value),ha='center',fontsize=10)
        baseline=data['carried_N48_aggregate']['conditions']['original'][role+'_correct'];ax.axhline(baseline,color='#29495d',linestyle='--',label='Carried native reference')
        ax.set_ylim(0,n+1.4);ax.set_xticks(x,labels);ax.set_ylabel('Correct first answers / '+str(n));ax.set_title(chr(69+col)+'. '+role.capitalize()+' | all six new conditions',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3)
    fig.text(.09,.025,'All 24 reused development contexts and both fixed donors retained. Recorded native P is not recomputed under hybrid R; F stays native to the recipient.\nComplete family/order contrasts and descriptive ranges remain in JSON. First full-vocabulary decisions, not full-program utility, meaning, mediation or fresh confirmation.',fontsize=10.5)
    fig.savefig(path,dpi=150);plt.close(fig)
