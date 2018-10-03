**** COMP 5421 Project1 - Intelligent Scissor ****
==================================================

Usage:
------
'''shell
cd bin
source activate
python main.py
'''

How to use?
-----------
### Load Image

> Click 'open file’ and select image file from file browser

### Draw Contour

> Click 'Draw Contour’ and use mouse to click on image, 'red cross’ will be shown as marker of your click

> Minimum Spanning Tree will be built, contour (shortest path) will be shown following your cursor

> Triple click to stop drawing

### Finish the contour

> Click 'Crop Image’ after stopped drawing 
> The contour will be completed by finding shortest path from the last click to the first click

