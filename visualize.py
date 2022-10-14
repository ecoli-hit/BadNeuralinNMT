from cProfile import label
import imp
from importlib.resources import path
from math import fabs
from os import mkdir
from sys import flags
from time import process_time_ns
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os.path

# up 

def up(path,out):
    if not os.path.isdir(out):
        mkdir(out)

    with open(path,'r') as fup:
        up = json.load(fp=fup)
    up_list = [v for k,v in up.items()]
    up_list = np.array(up_list)

    encoder_cnt = up_list[0:196].sum()
    decoder_cnt = up_list[196::].sum()
    total = encoder_cnt + decoder_cnt

    plt.bar(["encoder","decoder"],[float(encoder_cnt/total),float(decoder_cnt/total)], width=0.6)
    for a,b in zip(range(2),[float(encoder_cnt/total),float(decoder_cnt/total)]):
        plt.text(a, b+0.01, '%.2f'% (float(b)), ha='center', va= 'bottom',fontsize=10)
    plt.savefig(out+"/up1")


    norm_embedding = up_list[0:2].sum()
    layers = list()
    for i in range(0,12):
        layers.append(up_list[2+i*16:18+i*16].sum())
    layernorm_embedding = up_list[194:196].sum()
    x = np.append(np.append(norm_embedding/2048.,np.array(layers)/12591104.),layernorm_embedding/2048.)

    plt.figure(figsize=(25, 10), dpi=80)
    plt.bar(range(0, 14, 1),
            x, width=0.6)
    plt.xticks(list(range(0, 14, 1)), ["norm_embedding","layers.0","layers.1","layers.2","layers.3","layers.4","layers.5","layers.6","layers.7","layers.8","layers.9","layers.10","layers.11","layernorm_embedding"])
    for a,b in zip(range(0, 14, 1),x):
        plt.text(a, b+0.01, '%2f'% (float(b)), ha='center', va= 'bottom',fontsize=15)

    plt.savefig(out+"/up2")

    layer = list()
    for i in range(0,12):
        layer.append(list(up_list[2+i*16:18+i*16]))
    layer = np.array(layer)
    layer = layer.sum(axis=0)/12.
    k = [0,1024*1024,1024,1024*1024,1024,1024*1024,1024,1024*1024,1024,
         1024,1024,
         1024*4096,4096,
         4096*1024,1024,
         1024,1024]
    print(layer)
    layer = layer / k[1::]
    lab = ['self_attn_norm.k.weight','self_attn_norm.k.bias','self_attn_norm.v.weight','self_attn_norm.v.bias','self_attn_norm.q.weight','self_attn_norm.q.bias','self_attn_norm.out.weight','self_attn_norm.out.bias',
           'self_attn_norm.wieght','self_attn_norm.bias',
           'fc1.weight','fc1.bias','fc2.weight','fc2.bias',
           'final_layer_norm.weight','final_layer_norm.bias']

    plt.figure(figsize=(25, 10), dpi=80)
    plt.barh(range(16),
            layer, tick_label = lab)

    

    for a,b in zip(range(16),layer):
        plt.text(b+0.02,a , '%.2f'% (float(b)), ha='center', va= 'center',fontsize=15)
    plt.savefig(out+"/up3")





## down 


def down(path,out):
    if not os.path.isdir(out):
        mkdir(out)
        
    with open(path,'r') as fdown:
        down = json.load(fp=fdown) 

    down_list = [v for k,v in down.items()]
    down_list = np.array(down_list)

    encoder_cnt = down_list[0:196].sum()
    decoder_cnt = down_list[196::].sum()
    total = encoder_cnt+ decoder_cnt
    # encoder & decoder 
    plt.bar(["encoder","decoder"],[float( encoder_cnt) / total, float(decoder_cnt/total)], width=0.6)
    for a,b in zip(range(26),[float( encoder_cnt) / total, float(decoder_cnt/total)]):
        plt.text(a,b+0.01 , '%.2f'% (b), ha='center', va= 'bottom',fontsize=10)
    plt.savefig(out+"/down1")


    down_list = down_list[196::]
    norm_embedding = down_list[0:2].sum()
    layers = list()
    for i in range(0,12):
        layers.append(down_list[2+i*26:28+i*26].sum())
    layernorm_embedding = down_list[-2::].sum()
    x = np.append(np.append(norm_embedding,np.array(layers)/16796672.),layernorm_embedding)

    # decoder 分布
    plt.figure(figsize=(25, 10), dpi=80)
    plt.bar(range(1, 13, 1),
            x[1:13], width=0.6)
    plt.xticks(list(range(1, 13, 1)), ["layers.0","layers.1","layers.2","layers.3","layers.4","layers.5","layers.6","layers.7","layers.8","layers.9","layers.10","layers.11"])
    for a,b in zip(range(1, 13, 1),x[1:13]):
        plt.text(a, b+0.01, '%2f'% (float(b)) ,ha='center', va= 'bottom',fontsize=15)
    plt.savefig(out+"/down2")

    layer = list()
    for i in range(0,12):
        layer.append(list(down_list[2+i*26:28+i*26]))
    layer = np.array(layer)
    
    layer = layer.sum(axis=0)/12
    
    # print(layer.sum(axis=0))
    k = [0,1024*1024,1024,1024*1024,1024,1024*1024,1024,1024*1024,1024,1024,1024,
         1024*1024,1024,1024*1024,1024,1024*1024,1024,1024*1024,1024,1024,1024,
         1024*4096,4096,
         4096*1024,1024,
         1024,1024]
    layer = layer / k[1::]
    plt.figure(figsize=(25, 10), dpi=80)
    lab = ['self_attn_norm.k.weight','self_attn_norm.k.bias','self_attn_norm.v.weight','self_attn_norm.v.bias','self_attn_norm.q.weight','self_attn_norm.q.bias','self_attn_norm.out.weight','self_attn_norm.out.bias',
           'self_attn_norm.wieght','self_attn_norm.bias',
           'encoder_attn_norm.k.weight','encoder_attn_norm.k.bias','encoder_attn_norm.v.weight','encoder_attn_norm.v.bias','encoder_attn_norm.q.weight','encoder_attn_norm.q.bias','encoder_attn_norm.out.weight','encoder_attn_norm.out.bias',
           'encoder_attn_norm.wieght','encoder_attn_norm.bias',
           'fc1.weight','fc1.bias','fc2.weight','fc2.bias',
           'final_layer_norm.weight','final_layer_norm.bias']
    plt.barh(range(26),
            layer,tick_label=lab)

    
    for a,b in zip(range(26),layer):
        plt.text(b+0.01,a , '%.2f'% (b), ha='center', va= 'center',fontsize=10)
    plt.savefig(out+"/down3")
    print(layer.sum(axis=0)/12)

import sys

path = sys.argv[1]
out = sys.argv[2]
type = sys.argv[3]

if not len(sys.argv)==4:
    print("error params ")
    exit(0)

if type == "down":
    down(path,out)
elif type == "up":
    up(path,out)
    
    
