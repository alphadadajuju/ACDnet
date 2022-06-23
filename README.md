# ACDnet
The source codes (partial) of our article: ACDnet: An action detection network for real-time edge computing based on flow-guided feature approximation and memory aggregation (Pattern Recognition Letters: https://www.sciencedirect.com/science/article/abs/pii/S0167865521000568) have been released. 

Y. Liu, F. Yang, and D. Ginhac, “ACDnet: An Action Detection network for real-time edge computing based on flow-guided feature approximation and memory aggregation”, Pattern Recognition Letters, Special Issue on Advances on Human Action, Activity and Gesture Recognition (AHAAGR), Volume 145, 2021, Pages 118-126,

Note that we only release the parts related to constructing ACDnet's network architecture. This work was implemented prior to the middle of 2020 in MXNet (version 1.5 with their Symbol API) referring to the CVPR 2017 paper: Deep Feature Flow for Video Recognition; it will need to be adapted for more recent DL platforms/API. Codes for loading datasets and model evaluation can be referred to and adapted from https://github.com/zhreshold/mxnet-ssd.

# ACDnet Overview
We present a compact action detection network (**ACDnet**) targeting real-time edge computing which addresses both efficiency and accuracy. It intelligently exploits the temporal coherence between successive video frames to approximate their CNN features rather than naively extracting them. It also integrates memory feature aggregation from past video frames to enhance current detection stability, implicitly modeling long temporal cues over time.

* ACDnet concurrently addresses detection efficiency and accuracy. It combines feature approximation and memory aggregation modules, leading to improvement in both aspects.
* Our generalized framework allows for smooth integration with state-of-the-art detectors. When incorporated with SSD (single shot-detector), ACDnet could reason spatio-temporal context well over real-time, more appealing to resource-constrained devices.
* We conduct detailed studies in terms of accuracy, efficiency, robustness and qualitative analysis on public action datasets UCF-24 and JHMDB-21.

![alt text](https://github.com/alphadadajuju/ACDnet/blob/master/images/pipeline.jpg)

Experiments conducted on the public benchmark datasets UCF-24 and JHMDB-21 demonstrate that ACDnet, when integrated with the SSD detector, can robustly achieve detection well above real-time (75 FPS). At the same time, it retains reasonable accuracy (70.92 and 49.53 frame mAP) compared to other top-performing methods using far heavier conﬁgurations.

![alt text](https://github.com/alphadadajuju/ACDnet/blob/master/images/results.jpg)

# ACDnet Usage
