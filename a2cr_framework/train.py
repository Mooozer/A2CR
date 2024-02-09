# !/usr/bin/env python
# coding: utf-8

def findCommonMulti(worker, batch, lower = 1000, upper=1500):
    scm = (worker * batch) // gcd(worker, batch)
    while scm < lower:
        scm += scm 
    assert scm <= upper
    return scm



record = {'episode':[], 
          'training_episode_rewards': [], 
          'values': [], 
          'scores_test' : [],
          'frame_idxs' : [], 
          'critic_loss' : [], 
          'actor_loss' : [], 
          'total_loss': [], 
          'reasoner_loss': [],
          'elapsed_time':[],
          'rea_frame_idxs':[],
          'rea_labels':[]
         }

def training_and_saving(config_a2cr, model, model_reasoner):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config_a2cr['lr'], eps=1e-5)
    optimizer_reasoner = torch.optim.RMSprop(model_reasoner.parameters(), lr=config_a2cr['lr'], eps=1e-5)
    batch_memory = Batch_Memory()
    EXpool = ExploringPool(capacity=config_a2cr['Reasoner_capacity'])
    A2C_workers = []
    Rea_collector = []
    
    for _ in range(config_a2cr['A2C_workers']): A2C_workers.append(WorkerAndCollector(config_a2cr, model, labor='worker', w1 = config_a2cr['w1']))
    for _ in range(config_a2cr['Rea_collector']): Rea_collector.append(WorkerAndCollector(config_a2cr, model, labor='collector', w1 = config_a2cr['w1'] ))
    
    frame_idx, Rframe_idx = 0, 0 
    t0 = time.time()
    
    episode_reward = 0
    while frame_idx < config_a2cr['max_frames']:
        for worker in A2C_workers:
            states, actions, true_values  = worker.get_batch(EXpool, frame_idx=frame_idx)
            for i, _ in enumerate(states):
                batch_memory.pushWorkerData(
                    states[i],
                    actions[i],
                    true_values[i]
                )
            frame_idx += config_a2cr['batch_size']
        
        if frame_idx >= config_a2cr['Reasoner_Afterframes']:
            for collector in Rea_collector:
                diff_sbb, diff_vbb, rbb, pseudo_GTbb = collector.get_batch(EXpool, frame_idx=frame_idx)
                for i, _ in enumerate(states):
                    batch_memory.pushCollectorData(
                        diff_sbb[i],
                        diff_vbb[i],
                        rbb[i],
                        pseudo_GTbb[i]
                    )
                Rframe_idx += config_a2cr['batch_size']
               
        value, critic_loss, actor_loss, total_loss, reasoner_loss, pseudo_GTs = update(batch_memory, optimizer, optimizer_reasoner, frame_idx, config_a2cr)

        if frame_idx % config_a2cr['check_frames'] == 0:
            record['elapsed_time'].append(format_time(time.time() - t0))
            record['values'].append(value)
            record['frame_idxs'].append(frame_idx)
            record['critic_loss'].append(critic_loss)
            record['actor_loss'].append(actor_loss)
            record['total_loss'].append(total_loss)
            record['reasoner_loss'].append(reasoner_loss)
            record['rea_frame_idxs'].append(Rframe_idx)
            record['rea_labels'].append(pseudo_GTs)
            t1 = time.time()
            print(f"Elapsed: {format_time(t1 - t0)} Frames: {frame_idx}/{config_a2cr['max_frames']}, Values: {round(value,4)}, Critic: {round(critic_loss,4)}, Actor: {round(actor_loss,4)}, Total: {round(total_loss,4)}, Reasoner: {round(reasoner_loss,4)},ReaPoolSize: {len(EXpool.diffvpool)}")
            if config_a2cr['CI']:
                record['scores_test'].append(np.array([evaluate(config_a2cr['env_name'], model, model_reasoner, video=False, reasonSave=False, reasonSaveEvery = np.nan) if j!= config_a2cr['CI']-1 
                                          else evaluate(config_a2cr['env_name'], model, model_reasoner, video=False, reasonSave=False, 
                                                        reasonSaveEvery = np.nan) for j in range(config_a2cr['CI'])])) 
                print(f"Mean test rewards: {round(record['scores_test'][-1].mean(), 2)},  Std: {round(record['scores_test'][-1].std(), 2)}, ")
    
    #create path 
    saving_path = str(config_a2cr['env'].unwrapped) + "_entropy_" + str(config_a2cr['entropy_coef'])             
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    #save Training reward plot 
    plt.figure()              
    average_score = np.mean(record['training_episode_rewards'][-100:])
    plt.plot(record['episode'], record['training_episode_rewards'])
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Training Rewards')         
    plt.title(f"Total Frame: {record['frame_idxs'][-1]} \n Average Score (last 100 episode): {average_score}")
    plt.savefig(f"./{saving_path}/training_episode_rewards.png")

    #save Training values plot 
    plt.figure()              
    average_value = np.mean(record['values'][-1000:])
    plt.plot(record['frame_idxs'], record['values'])
    plt.grid()
    plt.xlabel('Frame')   
    plt.ylabel('Training Values')         
    plt.title(f"Total Frame: {record['frame_idxs'][-1]} \n Average Values (last 1000 frames): {average_value}")
    plt.savefig(f"./{saving_path}/training_episode_values.png")

                
    #save Training loss 
    plt.figure(figsize=(20, 12))
    ax = plt.subplot(221)
    plt.plot(record['frame_idxs'], record['critic_loss'])
    plt.title("Critic loss")
    plt.grid()
    plt.xlabel('Frame')   

    ax = plt.subplot(222)
    plt.plot(record['frame_idxs'], record['actor_loss'])
    plt.title("Actor loss")
    plt.grid()
    plt.xlabel('Frame')

    ax = plt.subplot(223)
    plt.plot(record['frame_idxs'], record['total_loss'])
    plt.title("Total loss")
    plt.grid()
    plt.xlabel('Frame')

    ax = plt.subplot(224)
    plt.plot(record['frame_idxs'], record['reasoner_loss'])
    plt.title("Reasoner loss")
    plt.grid()
    plt.xlabel('Frame')
    plt.savefig(f"./{saving_path}/loss.png")
    
    #save Test reward plot                  
    if config_a2cr['CI']:
        plt.figure()
        rd = pd.DataFrame((itertools.chain(*(itertools.product([record['frame_idxs'][i]], record['scores_test'][i]) 
                                             for i in range(len(record['frame_idxs']))))), columns=['Frames', 'Test Scores'])
        sns.lineplot(x="Frames", y="Test Scores", data=rd, ci='sd').set(title=f"Scores CI is based on {config_a2cr['CI']} episodes")
        plt.grid()                                                                 
        plt.savefig(f"./{saving_path}/test_episode_rewards.png")
                                                                        
    plt.close('all')
    #save record                 
    np.save(f"./{saving_path}/record.npy", record)

    #save networks
    torch.save(model.state_dict(), f"./{saving_path}/A2C_networks.pt")
    torch.save(model_reasoner.state_dict(), f"./{saving_path}/Reasoner_networks.pt")


 

env_name = 'SuperMarioBros-1-1-v0' #, 'SuperMarioBros-1-1-v0', 'SuperMarioBros-6-1-v0'
movement = 'COMPLEX_MOVEMENT' # RIGHT_ONLY, SIMPLE_MOVEMENT, and COMPLEX_MOVEMENT
env = Env_preprocess(env_name, movement = movement) 
test_env = Env_preprocess(env_name, movement = movement)

# env_name = 'SuperMarioBros-2-1-v0' #, 
# env = Env_preprocess(env_name) 
# test_env = Env_preprocess(env_name) 

########################
config_a2cr = {
    'env_name':env_name,
    'env': env,
    'test_env':test_env,
    'movement':movement,
    'gamma': 0.90, #0.99
    'max_frames' : 20000000,  #5000000
    'batch_size': 16, #5~20 
    'lr': 0.00025, #0.00025, #7e-4 if PongNoFrameskip-v4, 
    'entropy_coef': 0.5,
    'critic_coef': 0.5,
    'A2C_workers': 16,
    'Rea_collector':4,
    'CI': 0, # (0~inf, the CI is based on how many test games) 
    'Reasoner_CLS': 4, #number of reasoner class.
    'Reasoner_Afterframes': 10, #After how many frames, start updating the Reasoner Network
    'Reasoner_capacity':1000,
    'w1':0.5, #weight between delta_v and r in the process of obtaining label
}
config_a2cr['check_frames'] = findCommonMulti(worker=config_a2cr['A2C_workers'], batch=config_a2cr['batch_size'],
                                              lower = 1000, upper=1500)
saving_path = str(config_a2cr['env'].unwrapped) + "_entropy_" + str(config_a2cr['entropy_coef'])             
print("Current config_a2cr is:")
pprint(config_a2cr)



# model = A2CRModel(config_a2cr).to(device)
# model_reasoner = A2CRModel_Reasoner(config_a2cr).to(device)
# training_and_saving(config_a2cr, model, model_reasoner)