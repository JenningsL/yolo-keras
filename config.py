import numpy as np

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
origin_size = (1242, 375)
target_size = (416, 416)
C = len(CLASSES)
S = 13
batch_size = 8
anchor_boxes = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
B = len(anchor_boxes)
