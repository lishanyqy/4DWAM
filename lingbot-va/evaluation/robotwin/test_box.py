
from mplib.collision_detection.fcl import Box
import numpy as np
# tmp = Box(side=[np.float32(0.175), np.float32(0.05), np.float32(0.175)])
tmp = Box(side=np.array([0.35, 0.1, 0.35], dtype=np.float64))
print(tmp)