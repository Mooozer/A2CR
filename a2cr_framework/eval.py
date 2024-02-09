# !/usr/bin/env python
# coding: utf-8

import torch
import os, json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as T
import matplotlib.pyplot as plt

from lime import lime_image
from torch.autograd import Variable 
from torchvision import models, transforms 
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import mark_boundaries

#Grad-cam 
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def tensor2Img(diff_state):
    transform = T.ToPILImage()
    img = transform(diff_state)
    return img

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.CenterCrop(84)
    ])    
    return transf
pill_transf = get_pil_transform()


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        #normalize
    ])    
    return transf 
preprocess_transform = get_preprocess_transform()


    
def text_on_image(frame, action, reward, diff_value, stage_expl_value, reason, textSize=20):
    '''
    Add text on image
    '''
    font = ImageFont.truetype('./arial/arial.ttf', textSize) 
    im = Image.fromarray(frame)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer = ImageDraw.Draw(im)
    drawer.text((im.size[0]/20,im.size[1]/2), f'action: {action}', fill = text_color, font = font)
    drawer.text((im.size[0]/20,im.size[1]/2-textSize), f'reward: {reward}', fill = text_color, font = font)
    drawer.text((im.size[0]/20,im.size[1]/2-2*textSize), f'value diff: {round(diff_value.item(),2)}', fill = text_color, font = font)
    drawer.text((im.size[0]/20,im.size[1]/2-3*textSize), f'state exploration: {round(diff_value.item(),2)}', fill = text_color, font = font)
    drawer.text((im.size[0]/20,im.size[1]/2-4*textSize), f'reason: {reason}', fill = text_color, font = font)
    return im

def evaluate(config_a2cr_eval, model, r_model, saving_path, video=False, max_test_step = 100000, reasonSave = False, reasonSaveEvery=10, 
             Lime_Saliency = False, Grad_CAM_Saliency = False, only_record = True):
    '''
    config_a2cr_eval: config_a2cr for evaluating, should include env_name, movement, add_entropy, and add_entropy_times
    model: A2C network
    r_model: reasoner network
    saving_path: location for saving record / maps
    video: whether to show video
    max_test_step: maximum number of steps in a episode
    reasonSave: if save frame with reason text 
    reasonSaveEvery: How many frames are skipped per save (>=1)
    Lime_Saliency: if save Lime Saliency map
    Grad_CAM_Saliency: if save GradCAM Saliency map
    '''
    test_env = Env_preprocess(config_a2cr_eval['env_name'], movement = config_a2cr_eval['movement'])
    
    if reasonSave: 
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        path_i = 0    
        save_path = f'{saving_path}/episode{path_i}/'
        while os.path.exists(save_path):
            path_i += 1  
            save_path = f'{saving_path}/episode{path_i}/'
        os.mkdir(save_path)            

    evaluate_record = {'reason_output':[], 'reward_output':[], 'class_probs':[], 'actor_action':[],
                       'game_coins':[], 'game_life':[], 'game_score':[], 'game_time':[], 'x_pos':[], 'y_pos':[], 
                       'states': [],'stage_expl':[], 'diff_value':[],'diff_state':[],'action_ori_prob':[],'critic_value':[]}

    model.eval()
    r_model.eval()
    subspace_name = {0:'Breakout', 1:'Self-improvement', 2:'Hovering', 3:'Prospect'}

    state = test_env.reset()
    state = FloatTensor(state)
    reward_episode = 0
    done = False
    if video == False:
        t = 0 
        while not done and t< max_test_step:
            
            critic_value, actor_features = model.forward(state)
            evaluate_record['action_ori_prob'].append(actor_features.detach())
            evaluate_record['critic_value'].append(critic_value.detach())
            actor_features += config_a2cr_eval["add_entropy"] * config_a2cr_eval["add_entropy_times"] 
            actor_features = actor_features/torch.sum(actor_features) 
            dist = torch.distributions.Categorical(actor_features)
            action = dist.sample().item()
            #action = model.act(state)
            
            state_new, reward, done, info = test_env.step(action)

            stage_expl_value = stage_exploration(state, FloatTensor(state_new))
            evaluate_record['stage_expl'].append(stage_expl_value.item())
            diff_value = model.get_critic(FloatTensor(state_new)) - model.get_critic(state) 
            evaluate_record['diff_value'].append(diff_value.item())
            evaluate_record['reward_output'].append(reward)
            evaluate_record['actor_action'].append(action)
            evaluate_record['game_coins'].append(info['coins'])
            evaluate_record['game_life'].append(info['life'])
            evaluate_record['game_score'].append(info['score'])
            evaluate_record['game_time'].append(info['time'])
            evaluate_record['x_pos'].append(info['x_pos'])
            evaluate_record['y_pos'].append(info['y_pos'])
            evaluate_record['states'].append(state_new)
            evaluate_record['diff_state'].append(FloatTensor(state_new)-state)

            if reasonSave and t%reasonSaveEvery == 0:

                diff_state = FloatTensor(state_new) - state   
                r_cls_prob = r_model.forward(diff_state)

                evaluate_record['class_probs'].append(r_cls_prob.detach().view(-1))
                r_cls_name =  subspace_name[torch.argmax(r_cls_prob.view(-1)).item()]
                evaluate_record['reason_output'].append(r_cls_name)
                rgb_array = test_env.render(mode="rgb_array") 
                Image.fromarray(rgb_array).save(f'./{save_path}frame{t}.png')
                image = text_on_image(rgb_array, 
                                      action = [i for i in range(test_env.action_space.n)][action], 
                                      reward = reward, 
                                      diff_value = diff_value, 
                                      stage_expl_value = stage_expl_value, 
                                      reason = r_cls_name,
                                      textSize=20)
                #save frame:
                image.save(f'./{save_path}{t}.png')

            state = FloatTensor(state_new)
            reward_episode += reward            
            t+=1 

    elif video == True:
        t = 0 
        while not done and t< max_test_step: 
            critic_value, actor_features = model.forward(state)
            evaluate_record['action_ori_prob'].append(actor_features.detach())
            evaluate_record['critic_value'].append(critic_value.detach())

            actor_features += config_a2cr_eval["add_entropy"] * config_a2cr_eval["add_entropy_times"] 
            actor_features = actor_features/torch.sum(actor_features) 
            dist = torch.distributions.Categorical(actor_features)
            action = dist.sample().item()
            #action = model.act(state)
            
            state_new, reward, done, info = test_env.step(action)

            stage_expl_value = stage_exploration(state, FloatTensor(state_new))
            evaluate_record['stage_expl'].append(stage_expl_value.item())
            diff_value = model.get_critic(FloatTensor(state_new)) - model.get_critic(state) 
            evaluate_record['diff_value'].append(diff_value.item())
            evaluate_record['reward_output'].append(reward)
            evaluate_record['actor_action'].append(action)
            evaluate_record['game_coins'].append(info['coins'])
            evaluate_record['game_life'].append(info['life'])
            evaluate_record['game_score'].append(info['score'])
            evaluate_record['game_time'].append(info['time'])
            evaluate_record['x_pos'].append(info['x_pos'])
            evaluate_record['y_pos'].append(info['y_pos'])            
            evaluate_record['states'].append(state_new)
            evaluate_record['diff_state'].append(FloatTensor(state_new)-state)

            if reasonSave and t%reasonSaveEvery == 0:
                diff_state = FloatTensor(state_new) - state   
                r_cls_prob = r_model.forward(diff_state)
                evaluate_record['class_probs'].append(r_cls_prob.detach().view(-1))
                r_cls_index = torch.argmax(r_cls_prob.view(-1)).item()
                r_cls_name =  subspace_name[r_cls_index]
                evaluate_record['reason_output'].append(r_cls_name)
                rgb_array = test_env.render(mode="rgb_array")      

                if not only_record:
                    Image.fromarray(rgb_array).save(f'./{save_path}_frame{t}.png')
                    image = text_on_image(rgb_array,
                                          action = [i for i in range(test_env.action_space.n)][action], 
                                          reward = reward, 
                                          diff_value = diff_value, 
                                          stage_expl_value = stage_expl_value, 
                                          reason = r_cls_name,
                                          textSize=20)
                    image.save(f'./{save_path}{t}.png')

                ############################ Lime Saliency Map ############################
                if Lime_Saliency:
                    LimeS_image = tensor2Img(diff_state)
                    LimeS_image = pill_transf(LimeS_image)
                    
                    def batch_predict(images):
                        batch = torch.cat(tuple(preprocess_transform(i) for i in images), dim=0)                            
                        probs = r_model.forward(batch)
                        return probs.detach().cpu().numpy()
                    
                    explainer = lime_image.LimeImageExplainer()
                    
                    explanation = explainer.explain_instance(np.array(pill_transf(LimeS_image)), 
                                                             batch_predict, # classification function
                                                             top_labels=4, 
                                                             hide_color=0, 
                                                             batch_size = 1,
                                                             num_features = 5,
                                                             random_seed = 2023,
                                                             num_samples= 500 #size of the neighborhood to learn the linear model
                                                            )
                    
                    fig = plt.figure(figsize=(10,10))
                    f, axarr = plt.subplots(2,2)
                    
                    ax_ids = ((0,1),(1,1),(1,0),(0,0))
                    top_prob_index = torch.sort(r_cls_prob, descending=True).indices[0]
                    
                    for i in range(4):
                        class_index = top_prob_index[i]
                        temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], 
                                                                    positive_only=True, 
                                                                    num_features=5, #maximum number of features present in explanation
                                                                    hide_rest=True #make non-explanation part of the return image gray
                                                                   )
                        img_boundry2 = mark_boundaries(temp/255.0, mask)
                        LimeS =  np.array(diff_state).transpose(2,1,0) + np.dstack([mask.transpose()]*3)* 0.2 + img_boundry2.transpose(1,0,2)
                        LimeS = np.interp(LimeS, (LimeS.min(), LimeS.max()), (0,1))
                        axarr[ax_ids[i]].axis('off')
                        axarr[ax_ids[i]].imshow(LimeS) 
                        if class_index == r_cls_index:
                            title = axarr[ax_ids[i]].set_title(subspace_name[i])
                            title.set_color("orange")
                        else:
                            axarr[ax_ids[i]].title.set_text(subspace_name[i])

                    plt.subplots_adjust(wspace=-0.5, hspace=0.25) 
                    if not only_record:
                        plt.savefig(f'./{save_path}LimeSMap{t}.png')
                    plt.close('all')
                    
                ############################ Grad-CAM Saliency Map ############################

                if Grad_CAM_Saliency:
        
                    target_layers = [r_model.features_reasoner]
                    cam = GradCAM(model=r_model, target_layers=target_layers, use_cuda=False) #Construct the CAM object
                    
                    fig = plt.figure(figsize=(10,10))
                    
                    f, axarr = plt.subplots(2,2)
                    ax_ids = ((0,1),(1,1),(1,0),(0,0))
                    diff_state = diff_state.unsqueeze(0)
                    
                    for i in range(4):
                        targets = [ClassifierOutputTarget(i)]
                        grayscale_cam = cam(input_tensor=diff_state, targets=targets)
                        grayscale_cam = grayscale_cam[0,:,:]
                        axarr[ax_ids[i]].axis('off')
                        s_on_img = np.array(state[0,:,:]).transpose() +  grayscale_cam *0.5
                        axarr[ax_ids[i]].imshow(s_on_img) 
                        if i == r_cls_index:
                            title = axarr[ax_ids[i]].set_title(subspace_name[i])
                            title.set_color("orange")

                            #save sub plot method 1:
                            fig_sub = plt.figure(figsize=(10,10))
                            plt.imshow(s_on_img)
                            plt.axis('off')
                            fig_sub.savefig(f'./{save_path}GradCAM_SMap{t}_{subspace_name[r_cls_index]}.png')
                            plt.close() 

                        else:
                            axarr[ax_ids[i]].title.set_text(subspace_name[i])

                    plt.subplots_adjust(wspace=-0.5, hspace=0.25) 
                    if not only_record:
                        #save full plot
                        plt.savefig(f'./{save_path}GradCAM_SMap{t}.png')
                        #save sub plot method 2:
                        #extent = axarr[ax_ids[r_cls_index]].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        # plt.savefig(f'./{save_path}GradCAM_SMap{t}_{subspace_name[r_cls_index]}.png', bbox_inches=extent.expanded(1, 1))
                    plt.close('all')
                                

            state = FloatTensor(state_new)
            reward_episode += reward    
            test_env.render() 
            t+=1 

    test_env.close()

    evaluate_record['states'] = np.stack(evaluate_record['states'], axis=0)
    
    if reasonSave: 
        np.save(f"./{save_path}evaluate_record.npy", evaluate_record)
        
    print(f'frames: {t}, reward_episode: {reward_episode}')
    return t, reward_episode



env_name = 'SuperMarioBros-1-1-v0' #, 'SuperMarioBros-1-1-v0', 'SuperMarioBros-6-1-v0'
movement = 'COMPLEX_MOVEMENT' # RIGHT_ONLY, SIMPLE_MOVEMENT, and COMPLEX_MOVEMENT
env = Env_preprocess(env_name, movement = movement) 
test_env = Env_preprocess(env_name, movement = movement)

config_a2cr_eval = {
    'env_name':env_name,
    'env': env,
    'test_env':test_env,
    'movement':movement,
    'add_entropy' :0.001, 
    'add_entropy_times':0, #always 0, non-zero just for Entropy test 
    'Reasoner_CLS': 4}


A2CRModel_trained = A2CRModel(config_a2cr_eval).to(device)
A2CRModel_Reasoner_trained = A2CRModel_Reasoner(config_a2cr_eval).to(device)
saving_path = str(config_a2cr_eval['env'].unwrapped) + "/experiment1_1e6" #"/experiment_final_2e7"             
print(f'saving_path: {saving_path}')

A2CRModel_trained.load_state_dict(torch.load(f'./{saving_path}/A2C_networks.pt', 
                                             map_location=torch.device(device)))

A2CRModel_Reasoner_trained.load_state_dict(torch.load(f'./{saving_path}/Reasoner_networks.pt', 
                                                      map_location=torch.device(device)))

A2CRModel_trained.eval()
A2CRModel_Reasoner_trained.eval()

evaluate(config_a2cr_eval, A2CRModel_trained, A2CRModel_Reasoner_trained, saving_path,
         video=True,  max_test_step = 4000, reasonSave = True, reasonSaveEvery =1, 
         Lime_Saliency = False , Grad_CAM_Saliency = True, only_record=False) 




