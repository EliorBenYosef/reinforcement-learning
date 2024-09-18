import os
import tensorflow as tf
import torch
from tensorflow.python.keras.backend import backend as keras_tf_backend, set_session as keras_set_session
from tensorflow.python.client import device_lib as tf_device_lib

from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS


# DeviceGetUtils:

def tf_get_local_devices(GPUs_only=False):
    """
    Checks available devices to TensorFlow.
    :param GPUs_only:
    :return: a list of names of the devices that TensorFlow sees.
    """
    # assert 'GPU' in str(tf_device_lib.list_local_devices())

    local_devices = tf_device_lib.list_local_devices()  # local_device_protos
    # possible properties: name, device_type, memory_limit, incarnation, locality, physical_device_desc.
    #   name - str with the following structure: '/' + prefix ':' + device_type + ':' + device_type_order_num.
    #   device_type - CPU \ XLA_CPU \ GPU \ XLA_GPU
    #   locality (can be empty) - for example: { bus_id: 1 links {} }
    #   physical_device_desc (optional) - for example:
    #       "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
    if GPUs_only:
        return [dev.name for dev in local_devices if 'GPU' in dev.device_type]
    else:
        return [dev.name for dev in local_devices]


def keras_get_available_GPUs():
    """
    Checks available GPUs to Keras (>=2.1.1).
    :return:
    """
    # assert len(keras_tensorflow_backend._get_available_gpus()) > 0

    return keras_tf_backend._get_available_gpus()


def torch_get_current_device_name():
    """
    Checks available GPUs to PyTorch.
    :return:
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_device_name(torch.cuda.current_device())


# DeviceSetUtils:

def set_device(lib_type, devices_dict=None):
    """
    :param lib_type:
    :param devices_dict: {type: bus_id}. e.g.: {'CPU': 0}, {'GPU': 0, 'GPU': 1}
    :return:
    """
    # it seems that for:
    #   TF - tf_get_session_according_to_device() alone is enough...
    #   Keras - tf_set_device() alone is enough...
    # maybe only one method is enough for either?

    if devices_dict is not None:
        designated_GPUs_bus_id_str = ''
        for device_type, device_bus_id in devices_dict.items():
            if len(designated_GPUs_bus_id_str) > 0:
                designated_GPUs_bus_id_str += ','
            designated_GPUs_bus_id_str += str(device_bus_id)

        if lib_type == LIBRARY_TF:
            tf_set_device(designated_GPUs_bus_id_str)
        elif lib_type == LIBRARY_KERAS:
            tf_set_device(designated_GPUs_bus_id_str)
            keras_set_session_according_to_device(devices_dict)

# when trying to run a tensorflow \ keras model on GPU, make sure:
#   1. the system has a Nvidia GPU (AMD doesn't work yet).
#   2. the GPU version of tensorflow is installed.
#   3. CUDA is installed. https://www.tensorflow.org/install
#   4. tensorflow is running with GPU. use tf_get_local_devices()


def tf_set_device(designated_GPUs_bus_id_str):
    """
    :param designated_GPUs_bus_id_str: can be singular: '0', or multiple: '0,1'
    :return:
    """
    # set GPUs (CUDA devices) IDs' order by pci bus IDs (so it's consistent with nvidia-smi's output):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = designated_GPUs_bus_id_str  # specify which GPU ID(s) to be used.
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def tf_get_session_according_to_device(devices_dict):
    if devices_dict is not None:
        # allow_growth=True - limits session memory usage.
        #   starts with allocating an approximated amount of GPU memory, and expands if necessary:
        gpu_options = tf.GPUOptions(allow_growth=True)
        # # set the fraction of GPU memory to be allocated
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # log_device_placement - tells which device is used:
        config = tf.ConfigProto(device_count=devices_dict, gpu_options=gpu_options, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()
    return sess


def keras_set_session_according_to_device(devices_dict):
    # call this function after importing keras if you are working on a machine.
    keras_set_session(tf_get_session_according_to_device(devices_dict))
    # keras_tensorflow_backend.set_session(DeviceSetUtils.tf_get_session_according_to_device(device_map))


def torch_get_device_according_to_device_type(device_str):
    """
    :param device_str: e.g.: 'cpu', 'gpu', 'cuda:1'
    :return:
    """
    # enabling GPU vs CPU:
    if device_str == 'cpu':
        device = torch.device('cpu')  # default CPU. cpu:0 ?
    elif device_str == 'cuda:1':
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cuda')  # 2nd\default GPU. cuda:0 ?
    else:
        # default GPU \ default CPU (:0 ?):
        device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    return device
