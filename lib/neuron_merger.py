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
        return

    def copy_weights(self, model_a, model_b):
        """
        Copies weights from model_a to model_b.
        """
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        for index, layer in enumerate(state_dict_a):
            layer_shape_a = state_dict_a[layer].size()
            #print("layer:", layer, "len shape:", len(layer_shape_a))
            if(len(layer_shape_a) == 4):
                state_dict_b[layer] = state_dict_a[layer][:, :layer_shape_a[1], :, :]
            elif(len(layer_shape_a) == 2):
                state_dict_b[layer] = state_dict_a[layer][:, :layer_shape_a[1]]
            elif(len(layer_shape_a) == 1):
                state_dict_b[layer] = state_dict_a[layer][:layer_shape_a[0]]
            elif(len(layer_shape_a) == 0):
                state_dict_b[layer] = state_dict_a[layer]
        return

    def copy_weights_and_init(self, model_a, model_b):
        """
        Applies weight initialization and copies weights from model_a to model_b where possible
        """
        state_dict_b = model_b.state_dict()
        for _, layer in enumerate(state_dict_b):
            if(len(state_dict_b) == 4):
                torch.nn.init.xavier_uniform_(state_dict_b[layer])
        
        self.copy_weights(model_a, model_b)
        return

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
        self.copy_weights_and_init(model, expanded_model)
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

