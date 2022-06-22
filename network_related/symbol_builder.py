

import mxnet as mx
from symbol_.common import multi_layer_feature, multibox_layer
import numpy as np

def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)

#####
# modified FlowNetS
# input: key and non-key frame
# output: relative multi-scale flow fields (2-ch) and motion scale maps
# codes are referred from Deep Feature Flow
#####
def get_flownetS_dff(data):
    
    ### encoder
    flow_conv1 = mx.symbol.Convolution(name='f_flow_conv1', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
    ReLU1 = mx.symbol.LeakyReLU(name='f_ReLU1', data=flow_conv1 , act_type='leaky', slope=0.1)
    conv2 = mx.symbol.Convolution(name='f_conv2', data=ReLU1 , num_filter=128, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
    ReLU2 = mx.symbol.LeakyReLU(name='f_ReLU2', data=conv2 , act_type='leaky', slope=0.1)
    conv3 = mx.symbol.Convolution(name='f_conv3', data=ReLU2 , num_filter=256, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
    ReLU3 = mx.symbol.LeakyReLU(name='f_ReLU3', data=conv3 , act_type='leaky', slope=0.1)
    conv3_1 = mx.symbol.Convolution(name='f_conv3_1', data=ReLU3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    ReLU4 = mx.symbol.LeakyReLU(name='f_ReLU4', data=conv3_1 , act_type='leaky', slope=0.1)
    conv4 = mx.symbol.Convolution(name='f_conv4', data=ReLU4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
    ReLU5 = mx.symbol.LeakyReLU(name='f_ReLU5', data=conv4 , act_type='leaky', slope=0.1)
    conv4_1 = mx.symbol.Convolution(name='f_conv4_1', data=ReLU5 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    ReLU6 = mx.symbol.LeakyReLU(name='f_ReLU6', data=conv4_1 , act_type='leaky', slope=0.1)
    conv5 = mx.symbol.Convolution(name='f_conv5', data=ReLU6 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
    ReLU7 = mx.symbol.LeakyReLU(name='f_ReLU7', data=conv5 , act_type='leaky', slope=0.1)
    conv5_1 = mx.symbol.Convolution(name='f_conv5_1', data=ReLU7 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    ReLU8 = mx.symbol.LeakyReLU(name='f_ReLU8', data=conv5_1 , act_type='leaky', slope=0.1)
    conv6 = mx.symbol.Convolution(name='f_conv6', data=ReLU8 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
    ReLU9 = mx.symbol.LeakyReLU(name='f_ReLU9', data=conv6 , act_type='leaky', slope=0.1)
    conv6_1 = mx.symbol.Convolution(name='f_conv6_1', data=ReLU9 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    ReLU10 = mx.symbol.LeakyReLU(name='f_ReLU10', data=conv6_1 , act_type='leaky', slope=0.1)
    Convolution1 = mx.symbol.Convolution(name='f_Convolution1', data=ReLU10 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    deconv5 = mx.symbol.Deconvolution(name='f_deconv5', data=ReLU10 , num_filter=512, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_deconv5 = mx.symbol.Crop(name='f_crop_deconv5', *[deconv5,ReLU8] , offset=(1,1))
    ReLU11 = mx.symbol.LeakyReLU(name='f_ReLU11', data=crop_deconv5 , act_type='leaky', slope=0.1)
    
    ### decoder 
    upsample_flow6to5 = mx.symbol.Deconvolution(name='f_upsample_flow6to5', data=Convolution1 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='f_crop_upsampled_flow6_to_5', *[upsample_flow6to5,ReLU8] , offset=(1,1))
    Concat2 = mx.symbol.Concat(name='f_Concat2', *[ReLU8,ReLU11,crop_upsampled_flow6_to_5] )
    Convolution2 = mx.symbol.Convolution(name='f_Convolution2', data=Concat2 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    deconv4 = mx.symbol.Deconvolution(name='f_deconv4', data=Concat2 , num_filter=256, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_deconv4 = mx.symbol.Crop(name='f_crop_deconv4', *[deconv4,ReLU6] , offset=(1,1))
    ReLU12 = mx.symbol.LeakyReLU(name='f_ReLU12', data=crop_deconv4 , act_type='leaky', slope=0.1)
    upsample_flow5to4 = mx.symbol.Deconvolution(name='f_upsample_flow5to4', data=Convolution2 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='f_crop_upsampled_flow5_to_4', *[upsample_flow5to4,ReLU6] , offset=(1,1))
    Concat3 = mx.symbol.Concat(name='f_Concat3', *[ReLU6,ReLU12,crop_upsampled_flow5_to_4] )
    Convolution3 = mx.symbol.Convolution(name='f_Convolution3', data=Concat3 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    deconv3 = mx.symbol.Deconvolution(name='f_deconv3', data=Concat3 , num_filter=128, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_deconv3 = mx.symbol.Crop(name='f_crop_deconv3', *[deconv3,ReLU4] , offset=(1,1))
    ReLU13 = mx.symbol.LeakyReLU(name='f_ReLU13', data=crop_deconv3 , act_type='leaky', slope=0.1)
    upsample_flow4to3 = mx.symbol.Deconvolution(name='f_upsample_flow4to3', data=Convolution3 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_upsampled_flow4_to_3 = mx.symbol.Crop(name='f_crop_upsampled_flow4_to_3', *[upsample_flow4to3,ReLU4] , offset=(1,1))
    Concat4 = mx.symbol.Concat(name='f_Concat4', *[ReLU4,ReLU13,crop_upsampled_flow4_to_3] )
    Convolution4 = mx.symbol.Convolution(name='f_Convolution4', data=Concat4 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    deconv2 = mx.symbol.Deconvolution(name='f_deconv2', data=Concat4 , num_filter=64, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_deconv2 = mx.symbol.Crop(name='f_crop_deconv2', *[deconv2,ReLU2] , offset=(1,1))
    ReLU14 = mx.symbol.LeakyReLU(name='f_ReLU14', data=crop_deconv2 , act_type='leaky', slope=0.1)
    upsample_flow3to2 = mx.symbol.Deconvolution(name='f_upsample_flow3to2', data=Convolution4 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
    crop_upsampled_flow3_to_2 = mx.symbol.Crop(name='f_crop_upsampled_flow3_to_2', *[upsample_flow3to2,ReLU2] , offset=(1,1))
    Concat5 = mx.symbol.Concat(name='f_Concat5', *[ReLU2,ReLU14,crop_upsampled_flow3_to_2] )
    
    ### generate multi-scale flow fields and scale map
    # scale 0
    Concat5 = mx.symbol.Pooling(name='f_resize_concat5', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    Convolution5 = mx.symbol.Convolution(name='ft_Convolution5', data=Concat5 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False) 
    Convolution5_scale_bias = mx.sym.Variable(name='ft_Convolution5_scale_bias', lr_mult=0.0)
    Convolution5_scale = mx.symbol.Convolution(name='ft_Convolution5_scale', data=Concat5 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1),
                                               bias=Convolution5_scale_bias, no_bias=False)
    
    # scale 1
    Concat51 = mx.symbol.Pooling(name='ft_resize_concat51', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    Convolution51 = mx.symbol.Convolution(name='ft_Convolution51', data=Concat51 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    Convolution51_scale_bias = mx.sym.Variable(name='ft_Convolution51_scale_bias', lr_mult=0.0)
    Convolution51_scale = mx.symbol.Convolution(name='ft_Convolution51_scale', data=Concat51 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1),
                                               bias=Convolution51_scale_bias, no_bias=False)
    
    # scale 2
    Concat52 = mx.symbol.Pooling(name='ft_resize_concat52', data=Concat51 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    Convolution52 = mx.symbol.Convolution(name='ft_Convolution52', data=Concat52 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    Convolution52_scale_bias = mx.sym.Variable(name='ft_Convolution52_scale_bias', lr_mult=0.0)
    Convolution52_scale = mx.symbol.Convolution(name='ft_Convolution52_scale', data=Concat52 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1),
                                               bias=Convolution52_scale_bias, no_bias=False)
    
    # scale 3
    Concat53 = mx.symbol.Pooling(name='ft_resize_concat53', data=Concat52 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    Convolution53 = mx.symbol.Convolution(name='ft_Convolution53', data=Concat53 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    Convolution53_scale_bias = mx.sym.Variable(name='ft_Convolution53_scale_bias', lr_mult=0.0)
    Convolution53_scale = mx.symbol.Convolution(name='ft_Convolution53_scale', data=Concat53, num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1),
                                               bias=Convolution53_scale_bias, no_bias=False)
    
    # scale 4
    Concat54 = mx.symbol.Pooling(name='ft_resize_concat54', data=Concat53 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    Convolution54 = mx.symbol.Convolution(name='ft_Convolution54', data=Concat54 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    Convolution54_scale_bias = mx.sym.Variable(name='ft_Convolution54_scale_bias', lr_mult=0.0)
    Convolution54_scale = mx.symbol.Convolution(name='ft_Convolution54_scale', data=Concat54 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1),
                                               bias=Convolution54_scale_bias, no_bias=False)
    return mx.sym.Group([Convolution5, Convolution51, Convolution52, Convolution53, Convolution54, Convolution5_scale, Convolution51_scale, Convolution52_scale, Convolution53_scale, Convolution54_scale])

### Warp spatial features from a timestamp to another according to relative motion
### Used for TRAINING
def feat_warp_multiflow_witheqflag(layers, flow_orig, layers_dim, flow_scales, shape_flag, final_warp, eq_flag=None):
    
    select_conv_feats = []
    layer_name =  ['relu4_3', 'relu7', 'multi_feat_2_conv_3x3_relu', 'multi_feat_3_conv_3x3_relu', 'multi_feat_4_conv_3x3_relu']
    
    # Need to warp & scale each multi-scale feature with its corresponding flow field
    for l, from_layer in enumerate(layers):        
        
        flow = flow_orig[l] * flow_scales[l] 
            
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp')
        layer = mx.sym.BilinearSampler(data=layers[l], grid=flow_grid)
        
        # The below if-else condition only differs in layer naming
        if final_warp is False:
            layer = mx.sym.broadcast_mul(layer, flow_orig[l+5]) #name ="{}".format(from_name)
            layer = mx.sym.take(mx.sym.Concat(*[layer, layers[l]], dim=0), eq_flag, name = layer_name[l] + '_eq0')
        else:
            layer = mx.sym.broadcast_mul(layer, flow_orig[l+5]) #name ="{}".format(from_name)
            layer = mx.sym.take(mx.sym.Concat(*[layer, layers[l]], dim=0), eq_flag, name = layer_name[l])
    
        select_conv_feats.append(layer)
        
    return select_conv_feats

### Warp spatial features from a timestamp to another according to relative motion
### Used for TESTING
def feat_warp_inference_multiflow(layers, flow_orig, layers_dim, flow_scales, shape_flag, final_warp=False):
    select_conv_feats = []
    for l, from_layer in enumerate(layers):
        
        from_name = from_layer.name
        
        flow = flow_orig[l] * flow_scales[l]
        
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp')
        layer = mx.sym.BilinearSampler(data=layers[l], grid=flow_grid)
        
        # with scale map
        layer = mx.sym.broadcast_mul(layer, flow_orig[l+5],  name = "{}".format(from_name))
        
        select_conv_feats.append(layer)
    
    return select_conv_feats

### Adjust warped features' intensity according to flow
### Adopts similar notion/implementation as in FlowNet and Deep Feature Flow
def get_flow_scales(input_shape, layers_dim):
    flow_scales = []
    
    for l in range(len(layers_dim)):
        # only the first ii flow?
        if l <= 0: # 1
            scale = 1.0 / input_shape * layers_dim[l] * 20
            #scale =  1.0 / input_shape * layers_dim[l] * 40.0 # causing training to explode?
        else:
            scale = 1.0
        flow_scales.append(scale)
        
    return flow_scales

### Transform input features to a different latent space (for similarity measure)
def get_embednet(data, layer_num, feat_depth):
    layer_num = str(layer_num)
    em_conv1 = mx.symbol.Convolution(name='em_conv1_' + layer_num, data=data, num_filter=feat_depth/2, pad=(0, 0),
                                    kernel=(1, 1), stride=(1, 1), no_bias=False)
    em_ReLU1 = mx.symbol.Activation(name='em_ReLU1_'+ layer_num, data=em_conv1, act_type='relu')

    em_conv2 = mx.symbol.Convolution(name='em_conv2_'+ layer_num, data=em_ReLU1, num_filter=feat_depth/2, pad=(1, 1), kernel=(3, 3),
                                     stride=(1, 1), no_bias=False)
    em_ReLU2 = mx.symbol.Activation(name='em_ReLU2_'+ layer_num, data=em_conv2, act_type='relu')

    em_conv3 = mx.symbol.Convolution(name='em_conv3_'+ layer_num, data=em_ReLU2, num_filter=feat_depth*2, pad=(0, 0), kernel=(1, 1),
                                     stride=(1, 1), no_bias=False)

    return em_conv3


### Compute the importance weight of compared features based on cosine similiarity
def compute_weight(embed_flow, embed_conv_feat):
    embed_flow_norm = mx.symbol.L2Normalization(data=embed_flow, mode='channel')
    embed_conv_norm = mx.symbol.L2Normalization(data=embed_conv_feat, mode='channel')
    weight = mx.symbol.sum(data=embed_flow_norm * embed_conv_norm, axis=1, keepdims=True)

    return weight

# We refer to the codes of Deep Feature Flow and SSD
# The below code follows the THREE-frame training scheme as mentioned in our article
def get_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training ACDnet based on SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    # data_bef - data_aft (separated by T frames): simulates memory aggregation
    # data_aft - data (separated by 1,2, etc., up to T frames): simulates feature approximation]
    # only requires groundtruth of "data"
    
    data = mx.sym.Variable('data') 
    data_bef = mx.symbol.Variable("data_bef")
    data_aft = mx.symbol.Variable("data_aft")
    
    # special flags when data_bef == data_aft
    eq_flag = mx.symbol.Variable('eq_flag')
    
    # special flags when data_aft == data
    eq_flag_0 = mx.symbol.Variable('eq_flag_0')
    
    # groundtruth of "data"
    label = mx.sym.Variable('label') 
    
    
    ######
    ### 1. This entire block dedicates to feature approximation + memory aggregation
    ######
    
    # relative flow preparation (between data_bef & data_aft, and data_aft & data)
    dataf_concat1 = mx.symbol.Concat(data_aft / 255.0, data_bef / 255.0, dim=1)
    dataf_concat2 = mx.symbol.Concat(data / 255.0, data_aft / 255.0, dim=1)
    dataf_concat = mx.symbol.Concat(*[dataf_concat1, dataf_concat2], dim=0)
    flow = get_flownetS_dff(dataf_concat)
    
    flow1  = [] # data_bef & data_aft
    flow2 = [] # data_aft & data
    for l, from_layer in enumerate(flow):
        flow_slice  = mx.sym.SliceChannel(flow[l], axis=0, num_outputs=2)
        flow1.append(flow_slice[0]) # bef -> aft
        flow2.append(flow_slice[1]) # aft -> data
        
    layers_dim = [38, 19, 10, 5, 3] # 300x300
    shape_flag = 300
    
    # scale adjustment according to Deep Feature Flow and FlowNet's implementation
    flow_scales = get_flow_scales(shape_flag, layers_dim)
    
    # CNN feature extraction 
    concat_data = mx.symbol.Concat(*[data_bef, data_aft, data], dim=0)
    body = import_module(network).get_symbol(concat_data, num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=128)

    for k, from_layer in enumerate(layers):
        from_name = from_layer.name
        # normalize
        if normalizations[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_filters.pop(0), 1, 1), # num_channels.pop(0)
                init=mx.init.Constant(normalizations[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer, name=from_name)
        layers[k] = from_layer
    
    layers1  = []
    layers2 = []
    #layers_tar = []
    
    for l, from_layer in enumerate(layers):
       layers_slice  = mx.sym.SliceChannel(from_layer, axis=0, num_outputs=3, name= from_layer.name + '_slice')
       layers1.append(layers_slice[0]) # feat data_bef
       layers2.append(layers_slice[1]) # feat data_aft
       
       ## for standalone ssd
       #layers_tar.append(layers_slice[2])
       
    # warp bef to aft for feat alignment and fusion (memory aggregataion) 
    final_warp = False
    layers1_warped = feat_warp_multiflow_witheqflag(layers1, flow1, layers_dim, flow_scales, shape_flag, final_warp, eq_flag_0)
    
    weights1_all_layers  = [1]*5
    weights2_all_layers = [0]*5
    num_layer_to_fuse = 5
    adaptive_weight = True
    
    feat_depth  = [512,1024,512,256,256]
    
    layers2_fused = []
    for l in range (len(layers2)):
        if l < num_layer_to_fuse:
            concat_embed_data = mx.symbol.Concat(*[layers2[l], layers1_warped[l]], dim=0) # 16, 512,38, 38
            embed_output = get_embednet(concat_embed_data, l, feat_depth[l]) # 16, 1024, 38, 38
            embed_output = mx.sym.SliceChannel(embed_output, axis=0, num_outputs=2)
            unnormalize_weight1 = compute_weight(embed_output[0], embed_output[0]) # new extraction
            unnormalize_weight2 = compute_weight(embed_output[1], embed_output[0]) # impression
            
            unnormalize_weight1_ = mx.sym.expand_dims(unnormalize_weight1, axis = 0) # 8,1,38,38 -> 1,8,1,38,38
            unnormalize_weight2_ = mx.sym.expand_dims(unnormalize_weight2, axis = 0)
            unnormalize_weights = mx.symbol.Concat(unnormalize_weight1_, unnormalize_weight2_, dim=0) # 2,8,1,38,38
            
            weights = mx.symbol.softmax(data=unnormalize_weights, axis=0)
            weights = mx.sym.SliceChannel(weights, axis=0, num_outputs=2)
            
            weights1 = mx.sym.squeeze(weights[0], axis=0)
            weights2 = mx.sym.squeeze(weights[1], axis=0)
            weight1 = mx.symbol.tile(data=weights1, reps=(1, feat_depth[l], 1, 1))
            weight2 = mx.symbol.tile(data=weights2, reps=(1, feat_depth[l], 1, 1))
            weights1_all_layers[l] = weight1
            weights2_all_layers[l] = weight2
            
        if adaptive_weight is True: # weighted sum of bef and aft feat based on cosine similarity
            layers2_fused.append(layers2[l] * weights1_all_layers[l] + layers1_warped[l] * weights2_all_layers[l] )
    
        # for naive aggregation
        else:
            layers2_fused.append(layers2[l] * 0.5 + layers1_warped[l] * 0.5 )
    
    final_warp = True
    layers2_warp = feat_warp_multiflow_witheqflag(layers2_fused, flow2, layers_dim, flow_scales, shape_flag, final_warp, eq_flag)
    
    
    ######
    ### 2. This entire block dedicates to only feature approximation. 
    ######
    
    '''
    body = import_module(network).get_symbol(data_aft, num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter)
    
    for k, from_layer in enumerate(layers):
        from_name = from_layer.name
        # normalize
        if normalizations[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_filters.pop(0), 1, 1), # num_channels.pop(0)
                init=mx.init.Constant(normalizations[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer, name=from_name)
        layers[k] = from_layer
        
    dataf_concat = mx.symbol.Concat(data / 255.0, data_aft / 255.0, dim=1)
    
    # flow between key and non-key frame
    flow = get_flownetS_dff(dataf_concat)
    layers_dim = [38, 19, 10, 5, 3] # 300x300
    shape_flag = 300
    flow_scales = get_flow_scales(shape_flag, layers_dim)
    
    final_warp = True
    layers_warp = feat_warp_multiflow_witheqflag(layers, flow, layers_dim, flow_scales, shape_flag, final_warp, eq_flag)
    '''
    
    # The first argument to multibox_layer is either 1). layers2_warp (including both feature approximation and memory aggregation) 
    # or 2). layers_warp (only including feature approximation)
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers2_warp, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    
    tmp = mx.symbol.contrib.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]
    
    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.symbol.contrib.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    
    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    
    return out

#####
# Below codes are used for model INFERENCE
#####
    
### key frame (init) inference; functions as a stand-alone SSD
def get_symbol_dff_key_init(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    
    # data and data_key correspond to the same frame
    # data_key merely caches the current key frame for later use
    data = mx.symbol.Variable("data") 
    data_key = mx.sym.Variable("data_key")
    
    # SSD's multi-scale feature maps
    feat_key1 = mx.sym.Variable("relu4_3")
    feat_key2 = mx.sym.Variable("relu7")
    feat_key3 = mx.sym.Variable("multi_feat_2_conv_3x3_relu")
    feat_key4 = mx.sym.Variable("multi_feat_3_conv_3x3_relu")
    feat_key5 = mx.sym.Variable("multi_feat_4_conv_3x3_relu")
        
    body = import_module(network).get_symbol(data, num_classes=num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)
    
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    
    # returns key frame's multi-scale features + current key frame
    group = mx.sym.Group([out, data_key, feat_key1, feat_key2, feat_key3, feat_key4, feat_key5,
                              layers[0], layers[1], layers[2], layers[3], layers[4]])
    
    return group 

### other key frame inference; perform both memory aggregation and feature approximation
def get_symbol_dff_key(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    
    data = mx.symbol.Variable("data") # added by Alpha
    data_key = mx.sym.Variable("data_key")
    
    #data_imp = mx.sym.Variable("data_imp")
    
    feat_key1 = mx.sym.Variable("relu4_3")
    feat_key2 = mx.sym.Variable("relu7")
    feat_key3 = mx.sym.Variable("multi_feat_2_conv_3x3_relu")
    feat_key4 = mx.sym.Variable("multi_feat_3_conv_3x3_relu")
    feat_key5 = mx.sym.Variable("multi_feat_4_conv_3x3_relu")
        
    layers_imp = [feat_key1,feat_key2,feat_key3,feat_key4,feat_key5]
    
    # relative motion between current "key" and previous "key" frame (feature alignment for memory aggregation)
    dataf_concat = mx.symbol.Concat(data / 255.0, data_key / 255.0, dim=1)
    flow = get_flownetS_dff(dataf_concat)
    
    # extract features for current "key" frame
    body = import_module(network).get_symbol(data, num_classes=num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)
    
    layers_dim = [38, 19, 10, 5, 3] # 300x300
    shape_flag = 300
    
    # warp previous "key" frame feature to the current "key" frame's perspective
    flow_scales = get_flow_scales(shape_flag, layers_dim)
    layers_imp_warp = feat_warp_inference_multiflow(layers_imp, flow, layers_dim, flow_scales, shape_flag, final_warp=False)
    
    weights1_all_layers = [1]*5
    weights2_all_layers = [0]*5
    layers_fused = []
    feat_depth  = [512,1024,512,256,256]
    
    num_layer_to_fuse = 0
    adaptive_weight = True
    
    # compute importance weights for current key and warped key features
    for l in range (len(layers)):
        if l < num_layer_to_fuse:
            concat_embed_data = mx.symbol.Concat(*[layers[l], layers_imp_warp[l]], dim=0)
            embed_output = get_embednet(concat_embed_data, l, feat_depth[l])
            embed_output = mx.sym.SliceChannel(embed_output, axis=0, num_outputs=2)
            unnormalize_weight1 = compute_weight(embed_output[0], embed_output[0]) # new extraction
            unnormalize_weight2 = compute_weight(embed_output[1], embed_output[0]) # impression
            unnormalize_weights = mx.symbol.Concat(unnormalize_weight1, unnormalize_weight2, dim=0)
    
            weights = mx.symbol.softmax(data=unnormalize_weights, axis=0)
            weights = mx.sym.SliceChannel(weights, axis=0, num_outputs=2)
            weight1 = mx.symbol.tile(data=weights[0], reps=(1, feat_depth[l], 1, 1))
            weight2 = mx.symbol.tile(data=weights[1], reps=(1, feat_depth[l], 1, 1))
            weights1_all_layers[l] = weight1
            weights2_all_layers[l] = weight2
            
            ### debug: return weights to visualize
            #w1 = weights[0]
            #w2 = weights[1]
            
        if adaptive_weight is True:
            layers_fused.append(mx.sym.broadcast_add(mx.sym.broadcast_mul(layers[l], weights1_all_layers[l]), mx.sym.broadcast_mul(layers_imp_warp[l], weights2_all_layers[l]),  name=layers[l].name) )
        else: 
            layers_fused.append(layers[l])
            
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers_fused, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    
    group = mx.sym.Group([out, data_key, feat_key1, feat_key2, feat_key3, feat_key4, feat_key5,
                              layers_fused[0], layers_fused[1], layers_fused[2], layers_fused[3], layers_fused[4]])#, temporal_attention_debug[0], feat_imp_debug[0], feat_now_debug[0], feat_diff_debug[0]])
    
    return group 

### non-key frame inference; performs feature approximation
def get_symbol_dff_cur(network, num_classes, from_layers, num_filters, sizes, ratios,
       strides, pads, normalizations=-1, steps=[], min_filter=128,
       nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    
    data = mx.symbol.Variable("data") 
    data_ref = mx.symbol.Variable("data_key")
    
    layer1 = mx.symbol.Variable("relu4_3") 
    layer2 = mx.symbol.Variable("relu7") 
    layer3 = mx.symbol.Variable("multi_feat_2_conv_3x3_relu") 
    layer4 = mx.symbol.Variable("multi_feat_3_conv_3x3_relu") 
    layer5 = mx.symbol.Variable("multi_feat_4_conv_3x3_relu")
        
    layers = [layer1, layer2, layer3, layer4, layer5]
    
    # relative motion between current "non-key" and previous "key" frame (feature alignment for feature approximation)
    dataf_concat = mx.symbol.Concat(data / 255.0, data_ref / 255.0, dim=1)
    flow = get_flownetS_dff(dataf_concat)

    layers_dim = [38, 19, 10, 5, 3] # 300x300
        
    shape_flag = 300

    #flow_scales = [2.53, 1.27, 0.67, 0.33, 0.2]
    flow_scales = get_flow_scales(shape_flag, layers_dim)
    #flow_scales = [1., 1., 1., 1., 1.]
    
    # warp previous "key" frame feature to the current "non-key" frame's perspective
    layers_ = feat_warp_inference_multiflow(layers, flow, layers_dim, flow_scales, shape_flag, final_warp = True)
    
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers_, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    
    return out

