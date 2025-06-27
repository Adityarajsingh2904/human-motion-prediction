#!/usr/bin/env python
from utils.logging import setup_logging
setup_logging()
# encoding: utf-8
'''
@project : MSRGCN
@file    : __init__.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 16:34
'''
from .motiondataset import MotionDataset as CMUMotionDataset
