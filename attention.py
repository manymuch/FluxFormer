import numpy as np
import cv2

attention_weights = np.loadtxt("attention_weight.txt")
attention_day = attention_weights.mean(axis=0)
attention_day = attention_day.reshape(7, 8)

normed_attention = (attention_day - np.min(attention_day)) / (np.max(attention_day) - np.min(attention_day) + 1e-3)
# apply heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * normed_attention), cv2.COLORMAP_HOT)
# resize by scale 2
heatmap = cv2.resize(heatmap, (0, 0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
# write heatmap
cv2.imwrite("attention_heatmap.png", heatmap)
