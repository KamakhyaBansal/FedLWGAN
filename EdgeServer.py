import torch
#Try various combinations
'''
4: 0.8, 0.1, 0.1
5: [0.1,0.8,0.1]
6: [0.1,0.1,0.8]
'''
#model_weights = [weights(0.34,1.84), weights(0.29,2.03), weights(0.38,1.83)]
model_weights = [0.1,0.1,0.8]

files_list = [('/content/drive/MyDrive/my_gan_run/f1/model_2.pt'),'/content/drive/MyDrive/my_gan_run/f2/model_2.pt','/content/drive/MyDrive/my_gan_run/f3/model_2.pt']
def get_averaged_weights():
    #load average weights to main model

    gen_dict = torch.load(files_list[0])
    for i in range(0,len(model_weights)):
        for key in gen_dict['GAN'].keys() :
            if i == 0:
                gen_dict['GAN'][key] =  gen_dict['GAN'][key] * model_weights[0]
            else:
                gen_dict['GAN'][key] += model_weights[i] * torch.load(files_list[i])['GAN'][key]

    for key in gen_dict['GAN'].keys():
        gen_dict['GAN'][key] = gen_dict['GAN'][key]/sum(model_weights)

    torch.save(gen_dict,'/content/drive/MyDrive/my_gan_run/f1/model_6.pt')
get_averaged_weights()
