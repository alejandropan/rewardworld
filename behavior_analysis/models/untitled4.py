#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:56:14 2020

@author: alex
"""


def alternative(true_contrast, beliefSTD):
    bs_right=0
    for  x  in np.linspace(-1,1,100000):
        bs_right += norm.cdf(x,0,beliefSTD) * norm.pdf(x,true_contrast,beliefSTD) * (abs(-1 - 1)/100000)
    return [1-bs_right,bs_right]
