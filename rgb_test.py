# RGB Basler one frame

import numpy as np
import matplotlib.pyplot as plt

frame = np.fromfile('/home/lau/Documents/test_1/rock/r_1/Basler_a2A1920-160ucPRO__40648144__20260402_143142134_0000.raw', dtype=np.uint8)
frame = frame.reshape(1200, 1920)

plt.imshow(frame, cmap='gray')
plt.title('basler frame 0000')
plt.show()