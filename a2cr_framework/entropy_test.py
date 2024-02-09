# !/usr/bin/env python
# coding: utf-8

#HPC
env_range = range(1,9)
entropy_plus =  20
CI_times = 100

for Env_id in env_range: #environment 1~8
    env_name = f'SuperMarioBros-{Env_id}-1-v0'
    movement = 'COMPLEX_MOVEMENT' 
    env = Env_preprocess(env_name, movement = movement) 
    test_env = Env_preprocess(env_name, movement = movement)
    for entropy_time in range(0,entropy_plus):  #Entropy add times
        
        config_a2cr_eval = {
            'env_name':env_name,
            'env': env,
            'test_env':test_env,
            'movement':movement,
            'add_entropy' :0.001,
            'add_entropy_times': entropy_time, #always 0, just for Entropy test 
            'Reasoner_CLS': 4}

        A2CRModel_trained = A2CRModel(config_a2cr_eval).to(device)
        A2CRModel_Reasoner_trained = A2CRModel_Reasoner(config_a2cr_eval).to(device)
        saving_path = "Mario" + str(Env_id) + "/experiment_final_2e7"        

        A2CRModel_trained.load_state_dict(torch.load(f'./{saving_path}/A2C_networks.pt', 
                                             map_location=torch.device(device)))
        A2CRModel_Reasoner_trained.load_state_dict(torch.load(f'./{saving_path}/Reasoner_networks.pt', 
                                                      map_location=torch.device(device)))

        A2CRModel_trained.eval()
        A2CRModel_Reasoner_trained.eval()

        saving_path_inside = f'{saving_path}/entropy+{entropy_time}'
        if not os.path.exists(saving_path_inside):
            os.mkdir(saving_path_inside)   
        
        for k in range(CI_times): #How many times does CI based on
            evaluate(config_a2cr_eval, A2CRModel_trained, A2CRModel_Reasoner_trained, saving_path_inside,
                     video=True,  max_test_step = 1000, reasonSave = True, reasonSaveEvery =1, 
                     Lime_Saliency = False , Grad_CAM_Saliency = False, only_record=True) 

B_FreqDataDic = {**{'entropy+':range(0,entropy_plus)},
                 **{f'Env{i}_Rmean':[] for i in env_range}, 
                 **{f'Env{i}_Rstd':[] for i in env_range},
                 **{f'Env{i}_Rupper':[] for i in env_range},
                 **{f'Env{i}_Rlower':[] for i in env_range},
                 **{f'Env{i}_Rupper90':[] for i in env_range},
                 **{f'Env{i}_Rlower90':[] for i in env_range}}
S_FreqDataDic = {**{'entropy+':range(0,entropy_plus)},
                 **{f'Env{i}_Rmean':[] for i in env_range}, 
                 **{f'Env{i}_Rstd':[] for i in env_range},
                 **{f'Env{i}_Rupper':[] for i in env_range},
                 **{f'Env{i}_Rlower':[] for i in env_range},
                 **{f'Env{i}_Rupper90':[] for i in env_range},
                 **{f'Env{i}_Rlower90':[] for i in env_range}}
H_FreqDataDic = {**{'entropy+':range(0,entropy_plus)},
                 **{f'Env{i}_Rmean':[] for i in env_range}, 
                 **{f'Env{i}_Rstd':[] for i in env_range},
                 **{f'Env{i}_Rupper':[] for i in env_range},
                 **{f'Env{i}_Rlower':[] for i in env_range},
                 **{f'Env{i}_Rupper90':[] for i in env_range},
                 **{f'Env{i}_Rlower90':[] for i in env_range}}
P_FreqDataDic = {**{'entropy+':range(0,entropy_plus)},
                 **{f'Env{i}_Rmean':[] for i in env_range}, 
                 **{f'Env{i}_Rstd':[] for i in env_range},
                 **{f'Env{i}_Rupper':[] for i in env_range},
                 **{f'Env{i}_Rlower':[] for i in env_range},
                 **{f'Env{i}_Rupper90':[] for i in env_range},
                 **{f'Env{i}_Rlower90':[] for i in env_range}}

for Env_id in env_range:
    for entropy_time in range(0,entropy_plus):  #Entropy add times
        path1 = f'Mario{Env_id}/experiment_final_2e7/entropy+{entropy_time}'
        temp_B_list, temp_S_list, temp_P_list, temp_H_list = [], [], [], []
        for k in range(CI_times):
            path2 = f'{path1}/episode{k}/evaluate_record.npy'
            record = np.load(path2,allow_pickle=True).item()
            reasons = record['reason_output']
            B_ratio = sum(np.array(reasons) == 'Breakout')/len(reasons)
            S_ratio = sum(np.array(reasons) == 'Self-improvement')/len(reasons)
            P_ratio = sum(np.array(reasons) == 'Prospect')/len(reasons)
            H_ratio = sum(np.array(reasons) == 'Hovering')/len(reasons)
            temp_B_list.append(B_ratio), temp_S_list.append(S_ratio), temp_P_list.append(P_ratio), temp_H_list.append(H_ratio)
        
        B_FreqDataDic[f'Env{Env_id}_Rmean'].append(np.mean(temp_B_list))
        B_FreqDataDic[f'Env{Env_id}_Rstd'].append(np.std(temp_B_list))
        B_FreqDataDic[f'Env{Env_id}_Rupper'].append(np.quantile(temp_B_list, 0.975))
        B_FreqDataDic[f'Env{Env_id}_Rlower'].append(np.quantile(temp_B_list, 0.025))
        B_FreqDataDic[f'Env{Env_id}_Rupper90'].append(np.quantile(temp_B_list, 0.95))
        B_FreqDataDic[f'Env{Env_id}_Rlower90'].append(np.quantile(temp_B_list, 0.05))
        
        S_FreqDataDic[f'Env{Env_id}_Rmean'].append(np.mean(temp_S_list))
        S_FreqDataDic[f'Env{Env_id}_Rstd'].append(np.std(temp_S_list))
        S_FreqDataDic[f'Env{Env_id}_Rupper'].append(np.quantile(temp_S_list, 0.975))
        S_FreqDataDic[f'Env{Env_id}_Rlower'].append(np.quantile(temp_S_list, 0.025))
        S_FreqDataDic[f'Env{Env_id}_Rupper90'].append(np.quantile(temp_S_list, 0.95))
        S_FreqDataDic[f'Env{Env_id}_Rlower90'].append(np.quantile(temp_S_list, 0.05))
    
        H_FreqDataDic[f'Env{Env_id}_Rmean'].append(np.mean(temp_H_list))
        H_FreqDataDic[f'Env{Env_id}_Rstd'].append(np.std(temp_H_list))
        H_FreqDataDic[f'Env{Env_id}_Rupper'].append(np.quantile(temp_H_list, 0.975))
        H_FreqDataDic[f'Env{Env_id}_Rlower'].append(np.quantile(temp_H_list, 0.025))
        H_FreqDataDic[f'Env{Env_id}_Rupper90'].append(np.quantile(temp_H_list, 0.95))
        H_FreqDataDic[f'Env{Env_id}_Rlower90'].append(np.quantile(temp_H_list, 0.05))
        
        P_FreqDataDic[f'Env{Env_id}_Rmean'].append(np.mean(temp_P_list))
        P_FreqDataDic[f'Env{Env_id}_Rstd'].append(np.std(temp_P_list))
        P_FreqDataDic[f'Env{Env_id}_Rupper'].append(np.quantile(temp_P_list, 0.975))
        P_FreqDataDic[f'Env{Env_id}_Rlower'].append(np.quantile(temp_P_list, 0.025))
        P_FreqDataDic[f'Env{Env_id}_Rupper90'].append(np.quantile(temp_P_list, 0.95))
        P_FreqDataDic[f'Env{Env_id}_Rlower90'].append(np.quantile(temp_P_list, 0.05))
        
np.save(f"./B_FreqDataDic.npy", B_FreqDataDic)
np.save(f"./S_FreqDataDic.npy", S_FreqDataDic)
np.save(f"./H_FreqDataDic.npy", H_FreqDataDic)
np.save(f"./P_FreqDataDic.npy", P_FreqDataDic)

import pandas as pd
import numpy as np
B_evl_FreqData = pd.DataFrame(np.load('./Entropy_test_for_evaluation/B_FreqDataDic.npy',allow_pickle=True).item())
S_evl_FreqData = pd.DataFrame(np.load('./Entropy_test_for_evaluation/S_FreqDataDic.npy',allow_pickle=True).item())
H_evl_FreqData = pd.DataFrame(np.load('./Entropy_test_for_evaluation/H_FreqDataDic.npy',allow_pickle=True).item())
P_evl_FreqData = pd.DataFrame(np.load('./Entropy_test_for_evaluation/P_FreqDataDic.npy',allow_pickle=True).item())
all_evl_FreqData = [B_evl_FreqData, S_evl_FreqData, H_evl_FreqData, P_evl_FreqData]

from matplotlib.lines import Line2D
sns.set(rc={'figure.figsize':(10, 8)})
fig = plt.figure(figsize=(30,30))
f, axarr = plt.subplots(2,2)
ax_ids = ((0,1),(1,1),(1,0),(0,0))
subspace_name = {0:'Breakout', 1:'Self-improvement', 2:'Hovering', 3:'Prospect'}
for i in range(4):
    if i == 0 or i ==  3:
        axarr[ax_ids[i]].set_xticklabels([])
    for env_id in range(1,9):
        axarr[ax_ids[i]].plot(all_evl_FreqData[i]['entropy+'], 
                              all_evl_FreqData[i][f'Env{env_id}_Rmean'],  
                              label = f'v{env_id}', linewidth=2)  
        axarr[ax_ids[i]].fill_between(all_evl_FreqData[i]['entropy+'], 
                        all_evl_FreqData[i][f'Env{env_id}_Rmean'] - all_evl_FreqData[i][f'Env{env_id}_Rstd'], 
                        all_evl_FreqData[i][f'Env{env_id}_Rmean'] + all_evl_FreqData[i][f'Env{env_id}_Rstd'], 
                        alpha=0.1)
        
    if i == 1 or i == 2:
        axarr[ax_ids[i]].set_xlabel("Degree of entropy+",fontsize=20,labelpad=0)
        
    axarr[ax_ids[i]].set_ylabel(f"{subspace_name[i]} %",fontsize=20,labelpad=0)
    axarr[ax_ids[i]].tick_params(pad=-14,) #make ticks closer to plot

# leg1 = plt.legend(bbox_to_anchor=(1.08, 2.3), ncol=8, title = 'SuperMarioBros Environment', 
#                   title_fontsize = 15, fontsize=13)
leg1 = plt.legend(bbox_to_anchor=(1, 1.5), ncol=1, title = 'SuperMarioBros \n  Environment', 
                  title_fontsize = 15, fontsize=13)
plt.setp(axarr[ax_ids[2]].get_yticklabels()[1], visible=False)
plt.setp(axarr[ax_ids[2]].get_yticklabels()[7], visible=False)
plt.subplots_adjust(wspace=0.15, hspace=0.02) 


