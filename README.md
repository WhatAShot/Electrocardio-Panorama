# Electrocardio Panorama

This repository contains the code and datasets for our IJCAI 2021 paper *Electrocardio Panorama: Synthesizing New ECG views with Self-supervision*.

We propose a new concept called **Electrocardio Panorama**, which allows doctors to observe the ECG signals from any viewpoints and only requires one or few ECG views as input.

## Abstract
Multi-lead electrocardiogram (ECG) provides clinical information of heartbeats from several fixed viewpoints determined by the lead positioning. However, it is often not satisfactory to visualize ECG signals in these fixed and limited views, as some clinically useful information is represented only from a few specific ECG viewpoints. For the first time, we propose a new concept, Electrocardio Panorama, which allows visualizing ECG signals from any queried viewpoints. To build Electrocardio Panorama, we assume that an underlying electrocardio field exists, representing locations, magnitudes, and directions of ECG signals. We present a \textbf{N}eural \textbf{e}lectrocardio \textbf{f}ield \textbf{Net}work (Nef-Net), which first predicts the electrocardio field representation by using a sparse set of one or few input ECG views and then synthesizes Electrocardio Panorama based on the predicted representations. Specially, to better disentangle electrocardio field information from viewpoint biases, a new \textit{Angular Encoding} is proposed to process viewpoint angles. Also, we propose a self-supervised learning approach called \textit{Standin Learning}, which helps model the electrocardio field without direct supervision. Further, with very few modifications, Nef-Net can synthesize ECG signals from scratch. Experiments verify that our Nef-Net performs well on Electrocardio Panorama synthesis, and outperforms the previous work on the auxiliary tasks (ECG view transformation and ECG synthesis from scratch).

The code is just for reference, and is coming in these weeks.

Please cite the paper if the codes or dataset labels are helpful:

    @inproceedings{chen2021Electrocardio,
        author = {Chen, Jintai and Zheng, Xiangshang, and Yu, Hongyun and Chen, Danny Z and Wu, Jian},
        title = {{Electrocardio Panorama: Synthesizing New ECG views with Self-supervision}},
        booktitle = {IJCAI},
        year = {2021}
    }

(To be continued)
