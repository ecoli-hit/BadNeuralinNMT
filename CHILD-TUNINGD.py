from copy import copy, deepcopy
from distutils.command.clean import clean
from gettext import find
from importlib.resources import path
from select import select
from typing import Callable
from typing_extensions import Self
from torch.optim import Optimizer
from . import register_optimizer
from fairseq.optim import FairseqOptimizer
from typing import Callable, Iterable, Tuple
import math
import torch

@register_optimizer("CHILD_TUNINGD")
class CHILD_TUNINGD(FairseqOptimizer):
    def __init__(
        self, 
        args,
        params,
        ):
        
        super().__init__(args)
        
        self._optimizer = CTD(params)
        
    def optimizer_config(self):
        return super().optimizer_config

    
    
    
    
class CTD(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p = 1.0) -> None:
        self.maskdone = False
        
    
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.gradient_mask = None
        self.param_clean = None
        self.param_statistic_up   =None
        self.param_statistic_down =None
        super().__init__(params, defaults)
        
    def dump_file(self, path1, path2 ):
        import json
        f1= open(path1,'w')
        f2= open(path2,'w')
        json.dump(self.param_statistic_up,f1)
        json.dump(self.param_statistic_down,f2)
        f1.close()
        f2.close()
        
    def statu(self):
        i = 0
        print("group")
        # print(self.gradient_mask)
        
        for group in self.param_groups:
            for p in group["params"]:
                i += 1    
                print(p)

        
    def generate_mask(self):
        import numpy as np 
        import copy
        
        r = None
        gradient_mask = self.gradient_mask
        
        param_clean = self.param_clean
        
        
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        polar = np.percentile(r, (1-0.3)*100)
        lower = np.percentile(r,  0.1*100)
        
        up   = list()
        down = list()
        for k in gradient_mask:
            print(gradient_mask[k])
            gradient_mask[k] = gradient_mask[k] >= polar
            up.append(int((gradient_mask[k] == True).sum()))
        
        for k in param_clean:
            
            # print("down => {}".format(param_clean[k]))
            param_clean[k] = param_clean[k] >= lower
            down.append(int((param_clean[k] == False).sum()))

        
        mask_list = list()
        
        for k ,v  in gradient_mask.items():
            mask_list.append(v)
            
        clean_list = list()
        
        for k,v in param_clean.items():
            clean_list.append(v)
            
        self.gradient_mask = mask_list
        self.param_clean = clean_list
        
        i=0
        for k in self.param_statistic_up:
            print(up[i])
            self.param_statistic_up[k] = up[i]
            i+=1
        i=0
        for k in self.param_statistic_down:
            self.param_statistic_down[k] = down[i]
            i+=1
        
        print("------- done generate mask --------" )
        # print("mask rate {}".format(i / len(gradient_mask)))
        print('Polar => {}'.format(polar))
        print('Lower => {}'.format(lower))
        
        
        print("mute")
        i=0
        j=0
        for group in self.param_groups:
            for p in group["params"]:
                i+=1
                if i not in [1,2,199]:
                    p.data *= self.param_clean[j]
                    j+=1
                    # print("mute => {}".format(p.data))
        
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        i = 0
        j = 0
        for group in self.param_groups:
            for p in group["params"]:
                i+=1
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                if i not  in [1,2,199]:
                    grad *= self.gradient_mask[j]
                    j+=1
                    p.data *= self.param_clean[p]
                    
                    print("P mask=> {}".format(p))
                    
                    
                # if p in self.param_clean:
                #     p.data *= self.param_clean[p]
                #     print("in clean ")
                    


                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss