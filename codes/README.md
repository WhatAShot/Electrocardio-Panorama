## Datasets

Our proposed Nef-Net is performed on the [Tianchi dataset](https://tianchi.aliyun.com/competition/entrance/231754/information) and [PTB dataset](https://www.physionet.org/content/ptbdb/1.0.0/).

To use Tianchi dataset, one can download and use the data from the website or other mirrored links following the corresponding licence agreement.

To use the PTB dataset, one can downloaded [the pre-processed data](https://drive.google.com/file/d/1S6gNrIjtFH0WGjgsmEHNr4OgtDy9L3dS/view?usp=sharing), unzip and put them (2pkl files) into the `data/tianchi/npy_data/pkl_data/`.

## On annotations

In data pre-processing and the model processing, we require the signal (waveforms) segments and cardiac cycle annotations. The annotations are provided in json files, whose file name is similar to the original ECG data file but the filename extension.

In the annotation files, a cardiac cycle (heart beat) is splitted by the 6 breakpoints, whose keys are ["P on", "P off", "R on", "R off", "T on", "T off"] in the json files.

Note that in the annotation, we manage to keep the integrity of a cardiac cycle. That means, we do not provides the annotations for the incomplete cardiac cycles (typically the first and last ones) in a cases (one case contains many cardiac cycles). These cardiac cycles can be removed by some pre-processing.

## Model parameters

A model parameter file for reference can be [downloaded](https://drive.google.com/file/d/1tMTY-6LOxt1gSIn4jCi1BDO3EfL6CeOe/view?usp=sharing) and uploaded into  `codes/output/weight/nef_net/nef_net`. Then you can run an demo example (`demo.ipynb`).
