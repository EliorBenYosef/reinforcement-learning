
INPUT_TYPE_OBSERVATION_VECTOR = 0
INPUT_TYPE_STACKED_FRAMES = 1

########################

ATARI_FRAMES_STACK_SIZE = 4  # buffer_size. Number of consecutive frames. Gives the agent a sense of motion.

ATARI_IMAGE_CHANNELS_GRAYSCALE = 1
ATARI_IMAGE_CHANNELS_RGB = 3

########################

LIBRARY_TF = 0
LIBRARY_KERAS = 1
LIBRARY_TORCH = 2

########################

OPTIMIZER_Adam = 0
OPTIMIZER_RMSprop = 1
OPTIMIZER_Adadelta = 2
OPTIMIZER_Adagrad = 3
OPTIMIZER_SGD = 4

########################

NETWORK_TYPE_SEPARATE = 0
NETWORK_TYPE_SHARED = 1
