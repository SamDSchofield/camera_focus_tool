# Camera focus tool
Tool to help when focusing camera lenses

Calculates the Laplacian variance for the image and the specified ROIs. Higher variance = better focus.

## Usage
1. `rosrun camera_focus_tool camera_focus_tool.py --topic /your/image/topic`
2. Use rqt_reconfigure to shift the ROIs over a siemens star.
3. Open rqt plot and add the variance topics
4. Adjust the lens to increase the variance 


