# -*- coding: utf-8 -*-
"""
    chatnet
    ~~~~~~~

    package
"""
import logging

LOG = 'chatnet.log'
logFormatter = logging.Formatter('%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(20)
file_handler = logging.FileHandler(LOG)
file_handler.setLevel(20)
file_handler.setFormatter(logFormatter)
logger.addHandler(file_handler)
logger.info('logging initialized')
