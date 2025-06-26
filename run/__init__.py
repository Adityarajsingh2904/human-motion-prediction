#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# encoding: utf-8
from .h36m_runner import H36MRunner
from .cmu_runner import CMURunner