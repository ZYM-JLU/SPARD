import logging
import os
import torch


def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    return logger

class AverageMeterTorch(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_val = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        """
        val: [eval_batch_size], can contains nans
        """
        self.raw_val = val
        self.val = torch.nan_to_num(val)
        self.sum += sum(self.val).item()
        self.count += torch.count_nonzero(self.val).item()
        if self.count != 0:
            self.avg = self.sum / self.count

    def direct_set_avg(self, val):
        self.raw_val = val
        self.val = val
        self.avg = val
        self.sum = val
        self.count = 1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count