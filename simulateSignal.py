import numpy as np
import os
import pickle
module_path = os.path.dirname(__file__)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def simulate_signal(emissivity, projections_file_name):
    """Simulate the signals that would be expected with a line-of-sight approximation

    Parameters
    ----------
    emissivity: ndarray
        M x N array. Dimensions and coordinates must be coherent those defined in `projections.py`
    projections_file_name: str
        File name where the projections are saved. Do not add the extension at the end of the file name.

    Returns
    -------
    signals: ndarray
        1x32 array with the signals sequentially ordered

    Note:
        The signals are arbitrarily scaled and should be re-normalized
    """

    projections_dic = load_obj(os.path.join(module_path, projections_file_name))
    projections = projections_dic['projections']
    x = projections_dic['x']
    y = projections_dic['y']

    print('Projection coordinates x and y', x, y)

    p = projections.reshape((projections.shape[0], -1))
    signals = np.dot(p, emissivity.flatten())

    print('Output signals', signals)

    return signals
