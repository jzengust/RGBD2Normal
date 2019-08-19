####################################
# Load pre_trained model -- vgg_16 
# from torch model -- normal_scannet.t7
# Implemented by Jin Zeng, 20180720
#####################################
import torch
import torch.nn as nn
import vgg_16

def load_vgg_16(model, state='mp'):
    model_vgg = vgg_16.normal_scannet_cpu
    if state == 'scannet':
        model_vgg.load_state_dict(torch.load('./pre_trained/vgg_16.pth'))
    elif state == 'mp':
        model_vgg.load_state_dict(torch.load('./pre_trained/vgg_16_matterport.pth'))
    elif state == 'mp_in':
        model_vgg.load_state_dict(torch.load('./pre_trained/vgg_16_matterport.pth'))

    model_vgg_params = model_vgg.state_dict()
    model_params = model.state_dict()
    state = 'mp' # so that mp_in also got params loaded

    # copy parameter from model_vgg_params
    # conv1, conv+bn+conv+bn
    model_params['conv1.conv.0.weight'] = model_vgg_params['0.weight']
    model_params['conv1.conv.0.bias'] = model_vgg_params['0.bias']
    if state != 'mp_in':
        model_params['conv1.conv.1.weight'] = model_vgg_params['1.weight']
        model_params['conv1.conv.1.bias'] = model_vgg_params['1.bias']  
    # model_params['conv1.conv.1.running_mean'] = model_vgg_params['1.running_mean']
    # model_params['conv1.conv.1.running_var'] = model_vgg_params['1.running_var']
    model_params['conv1.conv.3.weight'] = model_vgg_params['3.weight']
    model_params['conv1.conv.3.bias'] = model_vgg_params['3.bias']
    if state != 'mp_in':
        model_params['conv1.conv.4.weight'] = model_vgg_params['4.weight']
        model_params['conv1.conv.4.bias'] = model_vgg_params['4.bias']  
    # model_params['conv1.conv.4.running_mean'] = model_vgg_params['4.running_mean']
    # model_params['conv1.conv.4.running_var'] = model_vgg_params['4.running_var']

    # conv2, conv+bn+conv+bn
    model_params['conv2.conv.0.weight'] = model_vgg_params['7.weight']
    model_params['conv2.conv.0.bias'] = model_vgg_params['7.bias']
    if state != 'mp_in':
        model_params['conv2.conv.1.weight'] = model_vgg_params['8.weight']
        model_params['conv2.conv.1.bias'] = model_vgg_params['8.bias']  
    # model_params['conv2.conv.1.running_mean'] = model_vgg_params['8.running_mean']
    # model_params['conv2.conv.1.running_var'] = model_vgg_params['8.running_var']
    model_params['conv2.conv.3.weight'] = model_vgg_params['10.weight']
    model_params['conv2.conv.3.bias'] = model_vgg_params['10.bias']
    if state != 'mp_in':
        model_params['conv2.conv.4.weight'] = model_vgg_params['11.weight']
        model_params['conv2.conv.4.bias'] = model_vgg_params['11.bias']  
    # model_params['conv2.conv.4.running_mean'] = model_vgg_params['11.running_mean']
    # model_params['conv2.conv.4.running_var'] = model_vgg_params['11.running_var']

    # conv3, conv+bn+conv+bn+conv+bn
    model_params['conv3.conv.0.weight'] = model_vgg_params['14.weight']
    model_params['conv3.conv.0.bias'] = model_vgg_params['14.bias']
    if state != 'mp_in':
        model_params['conv3.conv.1.weight'] = model_vgg_params['15.weight']
        model_params['conv3.conv.1.bias'] = model_vgg_params['15.bias']  
    # model_params['conv3.conv.1.running_mean'] = model_vgg_params['15.running_mean']
    # model_params['conv3.conv.1.running_var'] = model_vgg_params['15.running_var']
    model_params['conv3.conv.3.weight'] = model_vgg_params['17.weight']
    model_params['conv3.conv.3.bias'] = model_vgg_params['17.bias']
    if state != 'mp_in':
        model_params['conv3.conv.4.weight'] = model_vgg_params['18.weight']
        model_params['conv3.conv.4.bias'] = model_vgg_params['18.bias']  
    # model_params['conv3.conv.4.running_mean'] = model_vgg_params['18.running_mean']
    # model_params['conv3.conv.4.running_var'] = model_vgg_params['18.running_var']
    model_params['conv3.conv.6.weight'] = model_vgg_params['20.weight']
    model_params['conv3.conv.6.bias'] = model_vgg_params['20.bias']
    if state != 'mp_in':
        model_params['conv3.conv.7.weight'] = model_vgg_params['21.weight']
        model_params['conv3.conv.7.bias'] = model_vgg_params['21.bias']  
    # model_params['conv3.conv.7.running_mean'] = model_vgg_params['21.running_mean']
    # model_params['conv3.conv.7.running_var'] = model_vgg_params['21.running_var']

    # conv4, conv+bn+conv+bn+conv+bn
    model_params['conv4.conv.0.weight'] = model_vgg_params['24.weight']
    model_params['conv4.conv.0.bias'] = model_vgg_params['24.bias']
    if state != 'mp_in':
        model_params['conv4.conv.1.weight'] = model_vgg_params['25.weight']
        model_params['conv4.conv.1.bias'] = model_vgg_params['25.bias']  
    # model_params['conv4.conv.1.running_mean'] = model_vgg_params['25.running_mean']
    # model_params['conv4.conv.1.running_var'] = model_vgg_params['25.running_var']
    model_params['conv4.conv.3.weight'] = model_vgg_params['27.weight']
    model_params['conv4.conv.3.bias'] = model_vgg_params['27.bias']
    if state != 'mp_in':
        model_params['conv4.conv.4.weight'] = model_vgg_params['28.weight']
        model_params['conv4.conv.4.bias'] = model_vgg_params['28.bias']  
    # model_params['conv4.conv.4.running_mean'] = model_vgg_params['28.running_mean']
    # model_params['conv4.conv.4.running_var'] = model_vgg_params['28.running_var']
    model_params['conv4.conv.6.weight'] = model_vgg_params['30.weight']
    model_params['conv4.conv.6.bias'] = model_vgg_params['30.bias']
    if state != 'mp_in':
        model_params['conv4.conv.7.weight'] = model_vgg_params['31.weight']
        model_params['conv4.conv.7.bias'] = model_vgg_params['31.bias']  
    # model_params['conv4.conv.7.running_mean'] = model_vgg_params['31.running_mean']
    # model_params['conv4.conv.7.running_var'] = model_vgg_params['31.running_var']

    # conv5, conv+bn+conv+bn+conv+bn
    model_params['conv5.conv.0.weight'] = model_vgg_params['34.weight']
    model_params['conv5.conv.0.bias'] = model_vgg_params['34.bias']
    if state != 'mp_in':
        model_params['conv5.conv.1.weight'] = model_vgg_params['35.weight']
        model_params['conv5.conv.1.bias'] = model_vgg_params['35.bias']  
    # model_params['conv5.conv.1.running_mean'] = model_vgg_params['35.running_mean']
    # model_params['conv5.conv.1.running_var'] = model_vgg_params['35.running_var']
    model_params['conv5.conv.3.weight'] = model_vgg_params['37.weight']
    model_params['conv5.conv.3.bias'] = model_vgg_params['37.bias']
    if state != 'mp_in':
        model_params['conv5.conv.4.weight'] = model_vgg_params['38.weight']
        model_params['conv5.conv.4.bias'] = model_vgg_params['38.bias']  
    # model_params['conv5.conv.4.running_mean'] = model_vgg_params['38.running_mean']
    # model_params['conv5.conv.4.running_var'] = model_vgg_params['38.running_var']
    model_params['conv5.conv.6.weight'] = model_vgg_params['40.weight']
    model_params['conv5.conv.6.bias'] = model_vgg_params['40.bias']
    if state != 'mp_in':
        model_params['conv5.conv.7.weight'] = model_vgg_params['41.weight']
        model_params['conv5.conv.7.bias'] = model_vgg_params['41.bias']  
    # model_params['conv5.conv.7.running_mean'] = model_vgg_params['41.running_mean']
    # model_params['conv5.conv.7.running_var'] = model_vgg_params['41.running_var']

    # deconv5, conv+bn+conv+bn+conv+bn
    model_params['deconv5.conv.0.weight'] = model_vgg_params['43.weight']
    model_params['deconv5.conv.0.bias'] = model_vgg_params['43.bias']
    if state != 'mp_in':
        model_params['deconv5.conv.1.weight'] = model_vgg_params['44.weight']
        model_params['deconv5.conv.1.bias'] = model_vgg_params['44.bias']  
    # model_params['deconv5.conv.1.running_mean'] = model_vgg_params['44.running_mean']
    # model_params['deconv5.conv.1.running_var'] = model_vgg_params['44.running_var']
    model_params['deconv5.conv.3.weight'] = model_vgg_params['46.weight']
    model_params['deconv5.conv.3.bias'] = model_vgg_params['46.bias']
    if state != 'mp_in':
        model_params['deconv5.conv.4.weight'] = model_vgg_params['47.weight']
        model_params['deconv5.conv.4.bias'] = model_vgg_params['47.bias']  
    # model_params['deconv5.conv.4.running_mean'] = model_vgg_params['47.running_mean']
    # model_params['deconv5.conv.4.running_var'] = model_vgg_params['47.running_var']
    model_params['deconv5.conv.6.weight'] = model_vgg_params['49.weight']
    model_params['deconv5.conv.6.bias'] = model_vgg_params['49.bias']
    if state != 'mp_in':
        model_params['deconv5.conv.7.weight'] = model_vgg_params['50.weight']
        model_params['deconv5.conv.7.bias'] = model_vgg_params['50.bias']  
    # model_params['deconv5.conv.7.running_mean'] = model_vgg_params['50.running_mean']
    # model_params['deconv5.conv.7.running_var'] = model_vgg_params['50.running_var']

    # deconv4, conv+bn+conv+bn+conv+bn
    model_params['deconv4.conv.0.weight'] = model_vgg_params['52.weight']
    model_params['deconv4.conv.0.bias'] = model_vgg_params['52.bias']
    if state != 'mp_in':
        model_params['deconv4.conv.1.weight'] = model_vgg_params['53.weight']
        model_params['deconv4.conv.1.bias'] = model_vgg_params['53.bias']  
    # model_params['deconv4.conv.1.running_mean'] = model_vgg_params['53.running_mean']
    # model_params['deconv4.conv.1.running_var'] = model_vgg_params['53.running_var']
    model_params['deconv4.conv.3.weight'] = model_vgg_params['55.weight']
    model_params['deconv4.conv.3.bias'] = model_vgg_params['55.bias']
    if state != 'mp_in':
        model_params['deconv4.conv.4.weight'] = model_vgg_params['56.weight']
        model_params['deconv4.conv.4.bias'] = model_vgg_params['56.bias']  
    # model_params['deconv4.conv.4.running_mean'] = model_vgg_params['56.running_mean']
    # model_params['deconv4.conv.4.running_var'] = model_vgg_params['56.running_var']
    model_params['deconv4.conv.6.weight'] = model_vgg_params['58.weight']
    model_params['deconv4.conv.6.bias'] = model_vgg_params['58.bias']
    if state != 'mp_in':
        model_params['deconv4.conv.7.weight'] = model_vgg_params['59.weight']
        model_params['deconv4.conv.7.bias'] = model_vgg_params['59.bias']  
    # model_params['deconv4.conv.7.running_mean'] = model_vgg_params['59.running_mean']
    # model_params['deconv4.conv.7.running_var'] = model_vgg_params['59.running_var']

    # deconv3, conv+bn+conv+bn+conv+bn
    model_params['deconv3.conv.0.weight'] = model_vgg_params['61.weight']
    model_params['deconv3.conv.0.bias'] = model_vgg_params['61.bias']
    if state != 'mp_in':
        model_params['deconv3.conv.1.weight'] = model_vgg_params['62.weight']
        model_params['deconv3.conv.1.bias'] = model_vgg_params['62.bias']  
    # model_params['deconv3.conv.1.running_mean'] = model_vgg_params['62.running_mean']
    # model_params['deconv3.conv.1.running_var'] = model_vgg_params['62.running_var']
    model_params['deconv3.conv.3.weight'] = model_vgg_params['64.weight']
    model_params['deconv3.conv.3.bias'] = model_vgg_params['64.bias']
    if state != 'mp_in':
        model_params['deconv3.conv.4.weight'] = model_vgg_params['65.weight']
        model_params['deconv3.conv.4.bias'] = model_vgg_params['65.bias']  
    # model_params['deconv3.conv.4.running_mean'] = model_vgg_params['65.running_mean']
    # model_params['deconv3.conv.4.running_var'] = model_vgg_params['65.running_var']
    model_params['deconv3.conv.6.weight'] = model_vgg_params['67.weight']
    model_params['deconv3.conv.6.bias'] = model_vgg_params['67.bias']
    if state != 'mp_in':
        model_params['deconv3.conv.7.weight'] = model_vgg_params['68.weight']
        model_params['deconv3.conv.7.bias'] = model_vgg_params['68.bias']  
    # model_params['deconv3.conv.7.running_mean'] = model_vgg_params['68.running_mean']
    # model_params['deconv3.conv.7.running_var'] = model_vgg_params['68.running_var']

    # deconv2, conv+bn+conv+bn
    model_params['deconv2.conv.0.weight'] = model_vgg_params['70.weight']
    model_params['deconv2.conv.0.bias'] = model_vgg_params['70.bias']
    if state != 'mp_in':
        model_params['deconv2.conv.1.weight'] = model_vgg_params['71.weight']
        model_params['deconv2.conv.1.bias'] = model_vgg_params['71.bias']  
    # model_params['deconv2.conv.1.running_mean'] = model_vgg_params['71.running_mean']
    # model_params['deconv2.conv.1.running_var'] = model_vgg_params['71.running_var']
    model_params['deconv2.conv.3.weight'] = model_vgg_params['73.weight']
    model_params['deconv2.conv.3.bias'] = model_vgg_params['73.bias']
    if state != 'mp_in':
        model_params['deconv2.conv.4.weight'] = model_vgg_params['74.weight']
        model_params['deconv2.conv.4.bias'] = model_vgg_params['74.bias']  
    # model_params['deconv2.conv.4.running_mean'] = model_vgg_params['74.running_mean']
    # model_params['deconv2.conv.4.running_var'] = model_vgg_params['74.running_var']

    # deconv1, conv+conv
    model_params['deconv1.conv.0.weight'] = model_vgg_params['76.weight']
    model_params['deconv1.conv.0.bias'] = model_vgg_params['76.bias']
    model_params['deconv1.conv.2.weight'] = model_vgg_params['78.weight']
    model_params['deconv1.conv.2.bias'] = model_vgg_params['78.bias']

    return model_params
