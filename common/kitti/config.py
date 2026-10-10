import numpy as np

# [Unified] root_dir removed: dataset paths live in common/config.py


# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Back back (of vehicle) Point Cloud boundary for BEV

BEV_WIDTH = 608 # across y axis -25m ~ 25m
BEV_HEIGHT = 608 # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/BEV_HEIGHT


# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
	])

# cal mean from train set
R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
])


R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
#####################################################################################
# [Unified] Complex-YOLO does not regress z / height (paper definition).
# Values below are the fixed per-class priors used by the legacy
# inverse_yolo_target() (Complex_Yolo_Yolov3), kept verbatim.

# Legacy target builders inflate w / l by this margin (small objects);
# decoders subtract the same margin.
BOX_WL_MARGIN = 0.3
