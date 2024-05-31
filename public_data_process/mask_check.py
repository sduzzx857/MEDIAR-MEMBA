import cv2
import tifffile as tf

omni_path = '/home/data/MEDIAR/Public/lab/bthai_f_t001xy1_tile1_subtile1_cyto_masks.tiff'
DataScienceBowl2018_path = '/home/data/MEDIAR/Public/labels/0c90b86742b2.tiff'
live_cell_path = '/home/data/MEDIAR/Public/labels/A172_Phase_D7_1_01d04h00m_3.tiff'

omni_mask = tf.imread(omni_path)*255
DataScienceBowl2018_mask = tf.imread(DataScienceBowl2018_path)*255
live_cell_mask = tf.imread(live_cell_path)*255

cv2.imwrite('omni_mask.jpg', omni_mask)
cv2.imwrite('DataScienceBowl2018_mask.jpg', DataScienceBowl2018_mask)
cv2.imwrite('live_cell_mask.jpg', live_cell_mask)
