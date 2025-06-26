#!/usr/bin/env python
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
# encoding: utf-8
"""
@project : MSRGCN
@file    : __init__.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 16:34
"""

from .motiondataset import MotionDataset as H36MMotionDataset
