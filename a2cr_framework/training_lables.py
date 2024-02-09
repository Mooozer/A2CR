# !/usr/bin/env python
# coding: utf-8

#PLOT on HPC
config_a2cr = {'batch_size':16}
for ENV in range(3,9):
    
    training_record = np.load("./Mario"+str(ENV)+"/<SuperMarioBrosEnv<SuperMarioBros-"+str(ENV)+"-1-v0>>_entropy_0.5/record.npy",
                              allow_pickle=True).item()
    sns.set(rc={'figure.figsize':(8, 6)})
    from matplotlib.lines import Line2D
    import collections
    def frequency4(data):
        N = len(data)
        frequency = [v[1] for v in sorted(collections.Counter(data).items())]
        prob = np.array(list(frequency))/N
        return prob

    label0, label1, label2, label3, frames = [],[],[],[],[]
    training_labels = sum(training_record['rea_labels'],[])
    for i in range(len(training_labels)//1000):
        labels_chunk = training_labels[i*1000 : (i+1)*1000]
        a,b,c,d = frequency4(labels_chunk) 
        label0.append(a), label1.append(b), label2.append(c), label3.append(d)
        frames.append(i*1000*config_a2cr['batch_size'])
    data = pd.DataFrame({'B (1,1)':label0,'S (1,0)':label3, 'H (0,0)':label1, 'P (0,1)':label2, 'frames':frames})

    window_size = 40
    B_mean = data['B (1,1)'].rolling(window_size).mean()
    B_std = data['B (1,1)'].rolling(window_size).std()
    plt.plot(data['frames'], B_mean, color="red", label = 'B (1,1)')  
    plt.fill_between(data['frames'], B_mean - B_std, B_mean + B_std, color='red', alpha=0.3)

    S_mean = data['S (1,0)'].rolling(window_size).mean()
    S_std = data['S (1,0)'].rolling(window_size).std()
    plt.plot(data['frames'], S_mean, color="blue", label = 'S (1,0)')  
    plt.fill_between(data['frames'], S_mean - S_std, S_mean + S_std, color='blue', alpha=0.3)

    H_mean = data['H (0,0)'].rolling(window_size).mean()
    H_std = data['H (0,0)'].rolling(window_size).std()
    plt.plot(data['frames'], H_mean, color="black", label = 'H (0,0)')  
    plt.fill_between(data['frames'], H_mean - H_std, H_mean + H_std, color='black', alpha=0.3)

    P_mean = data['P (0,1)'].rolling(window_size).mean()
    P_std = data['P (0,1)'].rolling(window_size).std()
    plt.plot(data['frames'], P_mean, color="green", label = 'P (0,1)')  
    plt.fill_between(data['frames'], P_mean - P_std, P_mean + P_std, color='green', alpha=0.3)

    plt.ylabel('lable proportion',fontsize=16)
    plt.xlabel('training frames',fontsize=16)
    plt.title('Environment SuperMarioBros'+str(ENV),fontsize=16)
    leg1 = plt.legend(bbox_to_anchor=(1, 1), ncol=1)
    plt.savefig('./'+str(ENV)+'.png')
    plt.close('all')