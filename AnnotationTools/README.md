![The interface of the ECG annotation tool](https://steingate.oss-cn-hangzhou.aliyuncs.com/ui_main_window.png)

## Requirements

- Python
- PyQt5
- pyqtgraph
- tkinter

## Introduction

ECG interval labeling tool, used to label the starting and ending points of the P wave, QRS wave, and T wave of the ECG of the Tianchi dataset. 

## How to use ?

Click the "Open" button to open the ECG data to annotate. The interface will display the waveforms of lead II, lead V2, and lead V4 of the ECG. Click (using mouse) on the waveform in the interface, and the abscissa of the cursor will be shown in the upper right corner of the tool. Move the cursor to the point to be marked, and click the number 1-6 to mark:

Number 1: Starting point of P wave; 

Number 2: End point of P wave; 

Number 3: Starting point of QRS wave; 

Number 4: End point of QRS wave; 

Number 5: Starting point of T wave; 

Number 6: End point of T wave.

The annotation result will be saved in the file directory in json format. 
