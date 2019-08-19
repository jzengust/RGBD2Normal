from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F

def change_channel(outputs_norm, scale=1):
    row,col,channel=outputs_norm.shape
    nx = np.ones((row,col)) -outputs_norm[:, :, 0]
    ny = np.ones((row,col)) - outputs_norm[:, :, 1]
    nz = outputs_norm[:, :, 2]
    new_norm = [nx, nz, ny]
    return new_norm
def get_dataList(filename):
    f = open(filename, 'r')
    data_list = list()
    while 1:
        line = f.readline()
        line = line.strip()
        if (not line):
            break
        data_list.append(line)
    f.close()
    return data_list

def load_resume_state_dict(model, resume_state_dict):
     
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    resume_state_dict = {k: v for k, v in resume_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(resume_state_dict)

    return model_dict 

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]
        
def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module'
        new_state_dict[name] = v
    return new_state_dict

def norm_tf(outputs):
    bz, ch, img_rows, img_cols = outputs.size()
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)
    outputs_n = 0.5*(outputs_n+1)                
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n

def norm_sm(outputs):
    bz, ch, img_rows, img_cols = outputs.size()
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)              
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n

def norm_imsave(outputs):
    # outputs_s = np.squeeze(outputs.data.cpu().numpy(), axis=0)
    # outputs_s = outputs_s.transpose(1, 2, 0)
    # outputs_s = outputs_s.reshape(-1,3)
    # outputs_norm = sk.normalize(outputs_s, norm='l2', axis=1)
    # outputs_norm = outputs_norm.reshape(orig_size[0], orig_size[1], 3)
    # outputs_norm = 0.5*(outputs_norm+1)
    bz, ch, img_rows, img_cols = outputs.size()# bz should be one for imsave
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)
    outputs_n = 0.5*(outputs_n+1)                
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    # outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n

def get_fconv_premodel(model_F, resume_state_dict):
    model_params = model_F.state_dict()

    # copy parameter from resume_state_dict
    # conv1, conv+bn+conv+bn
    model_params['module.conv1.conv.0.weight'] = resume_state_dict['module.conv1.conv.0.weight']
    model_params['module.conv1.conv.0.bias'] = resume_state_dict['module.conv1.conv.0.bias']
    model_params['module.conv1.conv.1.weight'] = resume_state_dict['module.conv1.conv.1.weight']
    model_params['module.conv1.conv.1.bias'] = resume_state_dict['module.conv1.conv.1.bias']  
    model_params['module.conv1.conv.3.weight'] = resume_state_dict['module.conv1.conv.3.weight']
    model_params['module.conv1.conv.3.bias'] = resume_state_dict['module.conv1.conv.3.bias']
    model_params['module.conv1.conv.4.weight'] = resume_state_dict['module.conv1.conv.4.weight']
    model_params['module.conv1.conv.4.bias'] = resume_state_dict['module.conv1.conv.4.bias']  

    # conv2, conv+bn+conv+bn
    model_params['module.conv2.conv.0.weight'] = resume_state_dict['module.conv2.conv.0.weight']
    model_params['module.conv2.conv.0.bias'] = resume_state_dict['module.conv2.conv.0.bias']
    model_params['module.conv2.conv.1.weight'] = resume_state_dict['module.conv2.conv.1.weight']
    model_params['module.conv2.conv.1.bias'] = resume_state_dict['module.conv2.conv.1.bias']  
    model_params['module.conv2.conv.3.weight'] = resume_state_dict['module.conv2.conv.3.weight']
    model_params['module.conv2.conv.3.bias'] = resume_state_dict['module.conv2.conv.3.bias']
    model_params['module.conv2.conv.4.weight'] = resume_state_dict['module.conv2.conv.4.weight']
    model_params['module.conv2.conv.4.bias'] = resume_state_dict['module.conv2.conv.4.bias']  

    # conv3, conv+bn+conv+bn+conv+bn
    model_params['module.conv3.conv.0.weight'] = resume_state_dict['module.conv3.conv.0.weight']
    model_params['module.conv3.conv.0.bias'] = resume_state_dict['module.conv3.conv.0.bias']
    model_params['module.conv3.conv.1.weight'] = resume_state_dict['module.conv3.conv.1.weight']
    model_params['module.conv3.conv.1.bias'] = resume_state_dict['module.conv3.conv.1.bias']  
    model_params['module.conv3.conv.3.weight'] = resume_state_dict['module.conv3.conv.3.weight']
    model_params['module.conv3.conv.3.bias'] = resume_state_dict['module.conv3.conv.3.bias']
    model_params['module.conv3.conv.4.weight'] = resume_state_dict['module.conv3.conv.4.weight']
    model_params['module.conv3.conv.4.bias'] = resume_state_dict['module.conv3.conv.4.bias'] 
    model_params['module.conv3.conv.6.weight'] = resume_state_dict['module.conv3.conv.6.weight']
    model_params['module.conv3.conv.6.bias'] = resume_state_dict['module.conv3.conv.6.bias']
    model_params['module.conv3.conv.7.weight'] = resume_state_dict['module.conv3.conv.7.weight']
    model_params['module.conv3.conv.7.bias'] = resume_state_dict['module.conv3.conv.7.bias']   


    # # conv4, conv+bn+conv+bn+conv+bn
    # model_params['module.conv4.conv.0.weight'] = resume_state_dict['module.conv4.conv.0.weight']
    # model_params['module.conv4.conv.0.bias'] = resume_state_dict['module.conv4.conv.0.bias']
    # model_params['module.conv4.conv.1.weight'] = resume_state_dict['module.conv4.conv.1.weight']
    # model_params['module.conv4.conv.1.bias'] = resume_state_dict['module.conv4.conv.1.bias']  
    # model_params['module.conv4.conv.3.weight'] = resume_state_dict['module.conv4.conv.3.weight']
    # model_params['module.conv4.conv.3.bias'] = resume_state_dict['module.conv4.conv.3.bias']
    # model_params['module.conv4.conv.4.weight'] = resume_state_dict['module.conv4.conv.4.weight']
    # model_params['module.conv4.conv.4.bias'] = resume_state_dict['module.conv4.conv.4.bias'] 
    # model_params['module.conv4.conv.6.weight'] = resume_state_dict['module.conv4.conv.6.weight']
    # model_params['module.conv4.conv.6.bias'] = resume_state_dict['module.conv4.conv.6.bias']
    # model_params['module.conv4.conv.7.weight'] = resume_state_dict['module.conv4.conv.7.weight']
    # model_params['module.conv4.conv.7.bias'] = resume_state_dict['module.conv4.conv.7.bias']   

    # # conv5, conv+bn+conv+bn+conv+bn
    # model_params['module.conv5.conv.0.weight'] = resume_state_dict['module.conv5.conv.0.weight']
    # model_params['module.conv5.conv.0.bias'] = resume_state_dict['module.conv5.conv.0.bias']
    # model_params['module.conv5.conv.1.weight'] = resume_state_dict['module.conv5.conv.1.weight']
    # model_params['module.conv5.conv.1.bias'] = resume_state_dict['module.conv5.conv.1.bias']  
    # model_params['module.conv5.conv.3.weight'] = resume_state_dict['module.conv5.conv.3.weight']
    # model_params['module.conv5.conv.3.bias'] = resume_state_dict['module.conv5.conv.3.bias']
    # model_params['module.conv5.conv.4.weight'] = resume_state_dict['module.conv5.conv.4.weight']
    # model_params['module.conv5.conv.4.bias'] = resume_state_dict['module.conv5.conv.4.bias'] 
    # model_params['module.conv5.conv.6.weight'] = resume_state_dict['module.conv5.conv.6.weight']
    # model_params['module.conv5.conv.6.bias'] = resume_state_dict['module.conv5.conv.6.bias']
    # model_params['module.conv5.conv.7.weight'] = resume_state_dict['module.conv5.conv.7.weight']
    # model_params['module.conv5.conv.7.bias'] = resume_state_dict['module.conv5.conv.7.bias'] 

    # # deconv5, conv+bn+conv+bn+conv+bn
    # model_params['module.deconv5.conv.0.weight'] = resume_state_dict['module.deconv5.conv.0.weight']
    # model_params['module.deconv5.conv.0.bias'] = resume_state_dict['module.deconv5.conv.0.bias']
    # model_params['module.deconv5.conv.1.weight'] = resume_state_dict['module.deconv5.conv.1.weight']
    # model_params['module.deconv5.conv.1.bias'] = resume_state_dict['module.deconv5.conv.1.bias']  
    # model_params['module.deconv5.conv.3.weight'] = resume_state_dict['module.deconv5.conv.3.weight']
    # model_params['module.deconv5.conv.3.bias'] = resume_state_dict['module.deconv5.conv.3.bias']
    # model_params['module.deconv5.conv.4.weight'] = resume_state_dict['module.deconv5.conv.4.weight']
    # model_params['module.deconv5.conv.4.bias'] = resume_state_dict['module.deconv5.conv.4.bias'] 
    # model_params['module.deconv5.conv.6.weight'] = resume_state_dict['module.deconv5.conv.6.weight']
    # model_params['module.deconv5.conv.6.bias'] = resume_state_dict['module.deconv5.conv.6.bias']
    # model_params['module.deconv5.conv.7.weight'] = resume_state_dict['module.deconv5.conv.7.weight']
    # model_params['module.deconv5.conv.7.bias'] = resume_state_dict['module.deconv5.conv.7.bias'] 

    # # deconv4, conv+bn+conv+bn+conv+bn
    # model_params['module.deconv4.conv.0.weight'] = resume_state_dict['module.deconv4.conv.0.weight']
    # model_params['module.deconv4.conv.0.bias'] = resume_state_dict['module.deconv4.conv.0.bias']
    # model_params['module.deconv4.conv.1.weight'] = resume_state_dict['module.deconv4.conv.1.weight']
    # model_params['module.deconv4.conv.1.bias'] = resume_state_dict['module.deconv4.conv.1.bias']  
    # model_params['module.deconv4.conv.3.weight'] = resume_state_dict['module.deconv4.conv.3.weight']
    # model_params['module.deconv4.conv.3.bias'] = resume_state_dict['module.deconv4.conv.3.bias']
    # model_params['module.deconv4.conv.4.weight'] = resume_state_dict['module.deconv4.conv.4.weight']
    # model_params['module.deconv4.conv.4.bias'] = resume_state_dict['module.deconv4.conv.4.bias'] 
    # model_params['module.deconv4.conv.6.weight'] = resume_state_dict['module.deconv4.conv.6.weight']
    # model_params['module.deconv4.conv.6.bias'] = resume_state_dict['module.deconv4.conv.6.bias']
    # model_params['module.deconv4.conv.7.weight'] = resume_state_dict['module.deconv4.conv.7.weight']
    # model_params['module.deconv4.conv.7.bias'] = resume_state_dict['module.deconv4.conv.7.bias'] 

    return model_params

