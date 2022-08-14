# -*- coding: utf-8 -*-
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def concurrent_process(func, inputs, num_threads):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for batch_res in t.map(func, inputs):
            results.append(batch_res)
    return results


def init_logger(log_file=None):
    #
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step,info={'loss':20})
    '''

    def __init__(self, n_total, width=30, desc='Training',num_epochs = None):

        self.width = width
        self.n_total = n_total
        self.desc = desc
        self.start_time = time.time()
        self.num_epochs = num_epochs

    def reset(self):
        """Method to reset internal variables."""
        self.start_time = time.time()

    def _time_info(self, now, current):
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'
        return time_info

    def _bar(self, now, current):
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1: recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        return bar

    def epoch_start(self,current_epoch):
        sys.stdout.write("\n")
        if (current_epoch is not None) and (self.num_epochs is not None):
            sys.stdout.write(f"Epoch: {current_epoch}/{self.num_epochs}")
            sys.stdout.write("\n")

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        bar = self._bar(now, current)
        show_bar = f"\r{bar}" + self._time_info(now, current)
        if len(info) != 0:
            show_bar = f'{show_bar} ' + " [" + "-".join(
                [f' {key}={value:.4f} ' for key, value in info.items()]) + "]"
        if current >= self.n_total:
            show_bar += '\n'
        sys.stdout.write(show_bar)
        sys.stdout.flush()
