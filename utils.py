import numpy as np
import logging


def get_class_or_reg(task):
    if task in ['BBBP', 'BACE', 'ClinTox', 'Tox21', 'HIV', 'SIDER', 'MUV']:
        return "classification"
    if task in ['FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']:
        return "regression"


def find_indices(lst, value):
    return [index for index, element in enumerate(lst) if element == value]


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'./log/{name}.log')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def remove_indices(arr, indices):
    # 创建一个布尔索引数组，初始化为True
    keep_indices = np.ones(len(arr), dtype=bool)
    # 将指定的索引位置标记为False
    keep_indices[indices] = False
    # 使用布尔索引数组筛选需要保留的元素
    result = arr[keep_indices]
    return result
