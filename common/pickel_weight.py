import pickle

def save_weight(file_name:str, **kwargs):
    """保存模型权重到磁盘上

    Args:
        file_name (str): 保存权重的文件名，不包括文件扩展名
        **kwargs: 一个或多个参数，表示需要保存的权重

    Returns:
        None
    """
    with open(file_name + '_weight.pkl', 'wb') as f:
        pickle.dump(kwargs, f)


def load_weight(file_name:str) -> dict:
    """从磁盘上加载模型权重

    Args:
        file_name (str): 加载权重的文件名，不包括文件扩展名

    Returns:
        dict: 包含所有已加载的权重的字典对象
    """
    with open(file_name + '_weight.pkl', 'rb') as f:
        return pickle.load(f)