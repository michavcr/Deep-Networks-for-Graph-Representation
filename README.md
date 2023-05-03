# Deep-Networks-for-Graph-Representation
Implementation of a Deep Neural Network for learning Graph Representation (DNGR) with a PU learning further step to learn new graph edges.

This repository implements in Python (with the PyTorch library) the whole pipeline described in the article [1] which, in the context of drug discovery, consists in predicting new target-drugs interactions from a collection of known protein-protein (PPI) networks, drug similarities network and known in the literature drug-target interactions.

The implemented method is composed by two steps
- An embedding pipeline applied on each of the different types of network considered, that is based on random walk, pointwise mutual information and denoised autoencoders. This algorithm is a Graph Representation Learning technique especially developped in [2].
- A second step based on a recommander systems algorithm : aiming at finding unknown drug-target interactions with the previously computed embeddings (which represents either drugs or proteins and the connections they have with the other biological entities in the networks). This step is a "PU-Learning" step since these new relations must be learnt from Positive examples (already established drug-target association) and Unlabelled examples (nether established drug-target potential association).

In this repository :
- The `src/` directory contains the essentials of the implementation. `dngr.py` and `pu_learning.py` implements respectively both steps.
- The `notebook/` directory provides examples of utilization, in particular `Run on deepDTnet data.ipynb` runs the whole pipeline on the data used in [1]. 

### References

[1] Zeng, Xiangxiang & Zhu, Siyi & lu, Weiqiang & Huang, Jin & Liu, Zehui & Zhou, Yadi & Hou, Yuan & Huang, Yin & Guo, Huimin & Fang, Jiansong & Liu, Metheny & Trapp, Bruce & Li, Lang & Nussinov, Ruth & Eng, Charis & Loscalzo, Joseph & Cheng, Feixiong. (2019). Target Identification Among Known Drugs by Deep Learning from Heterogeneous Networks. SSRN Electronic Journal. 10.2139/ssrn.3385690.`

[2] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2016. Deep neural networks for learning graph representations. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI'16). AAAI Press, 1145–1152.
