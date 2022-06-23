# ACDnet
The source codes (partial) of our article: ACDnet: An action detection network for real-time edge computing based on flow-guided feature approximation and memory aggregation (Pattern Recognition Letters: https://www.sciencedirect.com/science/article/abs/pii/S0167865521000568) have been released. 

Y. Liu, F. Yang, and D. Ginhac, “ACDnet: An Action Detection network for real-time edge computing based on flow-guided feature approximation and memory aggregation”, Submitted to Pattern Recognition Letters, Special Issue on Advances on Human Action, Activity and Gesture Recognition (AHAAGR), Volume 145, 2021, Pages 118-126,

https://doi.org/10.1016/j.patrec.2021.02.001

Note that we only release the parts related to constructing ACDnet's network architecture. This work was implemented prior to the middle of 2020 in MXNet (version 1.5 with their Symbol API) referring to the CVPR 2017 paper: Deep Feature Flow for Video Recognition; it will need to be adapted for more recent DL platforms/API. Codes for loading datasets and model evaluation can be referred to and adapted from https://github.com/zhreshold/mxnet-ssd.
