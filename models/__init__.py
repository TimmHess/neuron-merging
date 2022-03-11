from .LeNet_300_100     import *
from .VGG       import *
from .ResNet    		import *
from .WideResNet		import *


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
    else:
        pass
    
    return model


def adjust_configuration(cfg, adjustment_percentage):
    for i in range(len(cfg)):
        if(type(cfg[i])==int): # and i > 3
            cfg[i] = int(cfg[i] * adjustment_percentage)
    #print("cfg: ", cfg)
    #temp_cfg = list(filter(('M').__ne__, cfg))
    #print("temp_cfg:", temp_cfg)
    return cfg