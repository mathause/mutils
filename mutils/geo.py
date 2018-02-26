#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 




def CosWgt(lat):
    """cosine-weighted latitude"""
	
	print('mutils.geo may deprecate')

	import numpy as np

    return np.cos(np.deg2rad(lat))






