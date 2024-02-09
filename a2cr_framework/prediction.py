# !/usr/bin/env python
# coding: utf-8

from matplotlib.lines import Line2D
evaluate_record = np.load(f"./<SuperMarioBrosEnv<SuperMarioBros-2-1-v0>>/env2_record_2e7_for_plot/evaluate_record.npy", 
                          allow_pickle=True).item()
pass_or_fail = 'fail'
evaluate_record_Data = pd.DataFrame({'State_Exploration': evaluate_record['stage_expl'], 
                                  'diff_value':evaluate_record['diff_value'], 
                                  'reward_output':evaluate_record['reward_output'], 
                                  'reason_output':evaluate_record['reason_output'],
                                  'Total_Gain': 0.5*np.array(evaluate_record['diff_value']) + 
                                                0.5*np.array(evaluate_record['reward_output']),
                                  'max_prob': [max(i).item() for i in evaluate_record['class_probs']]})
evaluate_record_Data.loc[evaluate_record_Data['max_prob'] < 0.1, 'reason_output'] = 'Uncertainty'

colors = {'Breakout': '#fa1505', 'Self-improvement': '#5b65f0', 
          'Hovering': 'black', 'Prospect':'#1fa130', 'Uncertainty':'yellow'}
sns.set(rc={'figure.figsize':(8, 8)})
g = sns.scatterplot(x="Total_Gain", y="State_Exploration",
              hue="reason_output",
              data=evaluate_record_Data, 
              alpha=0.5,
              palette = colors,
              s=100,
)
g.set_xlabel("Total Gain Value",fontsize=20)
g.set_ylabel("State Exploration Value",fontsize=20)

num_sub = 3
for reason in ['Breakout','Self-improvement','Hovering','Prospect']:
    SS = evaluate_record_Data[evaluate_record_Data['reason_output']==reason]['State_Exploration']
    SG = evaluate_record_Data[evaluate_record_Data['reason_output']==reason]['Total_Gain']
    Scutlines = [min(SS)+i*(max(SS)-min(SS))/num_sub for i in range(num_sub+1)]
    Gcutlines = [min(SG)+i*(max(SG)-min(SG))/num_sub for i in range(num_sub+1)]
    for x in Gcutlines:
        plt.plot([x, x], [min(SS), max(SS)], '--', lw=2,color=colors[reason], alpha = 0.2)
        plt.plot([x, x], [min(SS), max(SS)], '--', lw=2,color=colors[reason], alpha = 0.2)
    for y in Scutlines:
        plt.plot([min(SG), max(SG)], [y , y], '--', lw=2,color=colors[reason], alpha = 0.2)
        plt.plot([min(SG), max(SG)], [y , y], '--', lw=2,color=colors[reason], alpha = 0.2)
    
if pass_or_fail == 'fail':
    fail_point = len(evaluate_record['reason_output'])
    pass_point = random.randint(0, len(evaluate_record['reason_output'])-10)  

    #fail trajectory
    fail_sub_G, fail_sub_S = np.array([]), np.array([])
    for line in range(0,evaluate_record_Data.shape[0]):
        if line in range(fail_point-10, fail_point):
            fail_sub_G = np.append(fail_sub_G, [evaluate_record_Data['Total_Gain'][line]])
            fail_sub_S = np.append(fail_sub_S, [evaluate_record_Data['State_Exploration'][line]])
            if line!= fail_point-1:
                plt.text(evaluate_record_Data.Total_Gain[line], evaluate_record_Data.State_Exploration[line],
                      line, horizontalalignment='center', size='medium', color='black', weight='semibold')
            else:
                plt.text(evaluate_record_Data.Total_Gain[line], evaluate_record_Data.State_Exploration[line]-3,
                      "Failure", horizontalalignment='center', size='medium', color='black', weight='semibold')
    plt.plot(fail_sub_G[-1], fail_sub_S[-1], marker="s", markersize=15, markeredgecolor="black", markerfacecolor="black")   
    plt.quiver(fail_sub_G[:-1], fail_sub_S[:-1], fail_sub_G[1:]-fail_sub_G[:-1], fail_sub_S[1:]-fail_sub_S[:-1], 
               pivot = 'tail', scale_units='xy', angles='xy', scale=1.05, width=0.003, color='black')

#pass trajectory
for pass_trajectory in range(2): #how many trajectories
    pass_point = random.randint(0, len(evaluate_record['reason_output'])-10)  
    #if pass_trajectory ==0: pass_point = 11
    #if pass_trajectory ==1: pass_point = 100 

    pass_sub_G, pass_sub_S = np.array([]), np.array([])
    for line in range(0,evaluate_record_Data.shape[0]):
        if line in range(pass_point-10, pass_point):
            pass_sub_G = np.append(pass_sub_G, [evaluate_record_Data['Total_Gain'][line]])
            pass_sub_S = np.append(pass_sub_S, [evaluate_record_Data['State_Exploration'][line]])
            plt.text(evaluate_record_Data.Total_Gain[line], evaluate_record_Data.State_Exploration[line],
                      line, horizontalalignment='center', size='medium', color='black', weight='semibold')
    plt.quiver(pass_sub_G[:-1], pass_sub_S[:-1], pass_sub_G[1:]-pass_sub_G[:-1], pass_sub_S[1:]-pass_sub_S[:-1], 
               pivot = 'tail', scale_units='xy', angles='xy', scale=1.05, width=0.003, color='orange')

leg1 = g.legend(loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=1, fontsize=13)
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in ['black', 'orange']]
labels = ['Failure trajectory', 'Normal trajectory']
leg2 = plt.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.36, 0.29), fontsize=13)
plt.tick_params(pad=-5,)

plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)
plt.show()
