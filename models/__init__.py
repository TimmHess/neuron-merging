from .LeNet_300_100     import *
from .VGG               import *
from .ResNet    		import *
from .WideResNet		import *
from .SimpleCNN         import *

import copy


def save_state(model, acc, args):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            print(key)
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
                    
    # save
    if args.model_type == 'original':
        if args.arch == 'WideResNet' :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    '_'.join(map(str, args.depth_wide)),
                                    'pth.tar'])
        elif args.arch == 'ResNet' :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    str(args.depth_wide),
                                    'pth.tar'])
        else:
            model_filename = '.'.join([args.arch,
                                        args.dataset,
                                        args.model_type,
                                        'pth.tar'])
    else: # retrain
        if args.arch == 'WideResNet' :
            model_filename = '.'.join([args.arch,
                                    '_'.join(map(str, args.depth_wide)),
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])
        elif args.arch == 'ResNet' :
            model_filename = '.'.join([args.arch,
                                    str(args.depth_wide),
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])
        else :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])

    save_path = os.path.join('saved_models/', model_filename)
    torch.save(state, os.path.join('saved_models/', model_filename))
    print("Model saved at", save_path)
    return


def generate_model(arch_name:str, cfg, num_classes, args):
    model = None
    if args.arch == 'VGG':
        model = VGG(num_classes, cfg=cfg)
    elif args.arch == 'LeNet_300_100':
        model = LeNet_300_100(bias_flag=True, cfg=cfg)
    elif args.arch == 'ResNet':
        model = ResNet(int(args.depth_wide) ,num_classes,cfg=cfg)
    elif args.arch == 'WideResNet':
        model = WideResNet(args.depth_wide[0], num_classes, widen_factor=args.depth_wide[1], cfg=cfg)
    elif args.arch == 'SimpleCNN':
        model = SimpleCNN(num_classes, cfg=cfg)
    else:
        pass
    
    return model


def adjust_configuration(cfg, adjustment_percentage):
    tmp_cfg = copy.deepcopy(cfg)
    for i in range(len(tmp_cfg)):
        if(type(tmp_cfg[i])==int):
            tmp_cfg[i] = int(tmp_cfg[i] * adjustment_percentage)
    #print("cfg: ", cfg)
    #temp_cfg = list(filter(('M').__ne__, cfg))
    #print("temp_cfg:", temp_cfg)
    return tmp_cfg


def weight_init(model, decomposed_weight_list):
    for layer in model.state_dict():
        decomposed_weight = decomposed_weight_list.pop(0)
        model.state_dict()[layer].copy_(decomposed_weight)
    return model