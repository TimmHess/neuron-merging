from email.mime import base
import torch
import torch.nn as nn

from lib.decompose import Decompose

import models

import sys

class NeuronMerger():
    def __init__(self, expand_percentage:int, args) -> None:
        self.expand_percentage = expand_percentage
        self.args = args
        self.base_cfg = None

        self.base_weight_shapes = {} # storing the non-expanded model's weight dimension
        return

    def copy_weights(self, model_a, model_b):
        """
        Copies weights from model_a to model_b.
        """
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        for index, layer in enumerate(state_dict_a):
            #state_dict_b[layer] = state_dict_a[layer]
            layer_shape_a = state_dict_a[layer].size()
            if(len(layer_shape_a) == 4):
                state_dict_b[layer][:layer_shape_a[0], :layer_shape_a[1], :, :] = state_dict_a[layer]
            elif(len(layer_shape_a) == 2):
                state_dict_b[layer][:layer_shape_a[0], :layer_shape_a[1]] = state_dict_a[layer]
            elif(len(layer_shape_a) == 1):
                state_dict_b[layer][:layer_shape_a[0]] = state_dict_a[layer]
            elif(len(layer_shape_a) == 0):
                state_dict_b[layer] = state_dict_a[layer]
        return state_dict_b


    def expand(self, model:nn.Sequential):
        """
        Expands each layer of the provided model by the given expand_percentage.
        """
        self.base_cfg = model.cfg
        # Grab num classes from model
        num_classes = 10 # TODO:
        # Adjust configuration (cfg) for expanded model
        exp_cfg = models.adjust_configuration(model.cfg, adjustment_percentage=(1+self.expand_percentage))
        # Generate expanded model
        expanded_model = models.generate_model(self.args.arch, exp_cfg, num_classes, self.args)
        if(self.args.cuda):
            expanded_model.cuda()
        # Copy model weights 
        expanded_model.load_state_dict(self.copy_weights(model, expanded_model))
        return expanded_model


    def compress(self, model:nn.Sequential):
        """
        Compresses each layer of the provided model by the given expand_percentage.
        """
        # Grab num classes from model
        num_classes = 10 #TODO:
        # Generate compressed model
        compressed_model = models.generate_model(self.args.arch, self.base_cfg, num_classes, self.args)
        if(self.args.cuda):
            compressed_model.cuda()
        # Run the neuron-merging algorithm
        tmp_comp_cfg = list(filter(('M').__ne__, self.base_cfg)) # removes 'unimportant' configuration parameters (e.g. indications for pooling layers)
        decomposed_weights = Decompose(self.args.arch, model.state_dict(), self.args.criterion, self.args.threshold, 
                            self.args.lamda, self.args.model_type, tmp_comp_cfg, self.args.cuda, self.args.no_bn).main()
        # Store 'decomposed' weights back to compressed model
        compressed_model = models.weight_init(compressed_model, decomposed_weights)
        return compressed_model

    def compress_deterministic(self, model:nn.Sequential):
        num_classes = 10
        compressed_model = models.generate_model(self.args.arch, self.base_cfg, num_classes, self.args)
        if(self.args.cuda):
            compressed_model.cuda()
        state_dict_curr = model.state_dict()
        state_dict_comp = compressed_model.state_dict()
        for layer in state_dict_comp:
            comp_layer_shape = state_dict_comp[layer].size()
            if len(state_dict_comp[layer].size()) == 4:
                state_dict_comp[layer] = state_dict_curr[layer][:comp_layer_shape[0], :comp_layer_shape[1]]
            elif len(state_dict_comp[layer].size()) == 1:
                state_dict_comp[layer] = state_dict_curr[layer][:comp_layer_shape[0]]
        compressed_model.load_state_dict(state_dict_comp)
        return compressed_model


    def store_weight_shapes(self, model:nn.Sequential):
        model_state_dict = model.state_dict()
        for _, layer in enumerate(model_state_dict):
            self.base_weight_shapes[layer] = model_state_dict[layer].size()
        return


    def freeze_old_weights(self, model:nn.Sequential):
        """
        Intended to be called after backwards and before optimizer step to cancel gradient updates
        for all neurons of the base model. 
        This implies model weight updates are only happening by neuron merging
        """
        #print(model.feature[0].weight.grad)
        #grad_dict = {k:v.grad for k,v in model.named_parameters()}
        #for key, v in model.named_parameters():
        #    print(key)
        #sys.exit()
        #for i, layer in enumerate(model.state_dict()):
        #    print(i, layer)
        #    print(grad_dict[layer])
    
        model_state_dict = model.state_dict()
        for key, value in model.named_parameters():
            #value.grad.zero_()
            base_shape = self.base_weight_shapes[key]
            if len(base_shape) == 4:
                value.grad[:base_shape[0], :base_shape[1], :, :].zero_()
            elif len(base_shape) == 1:
                value.grad[:base_shape[0]].zero_()
            else:
                raise NotImplementedError
        return