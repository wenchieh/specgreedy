Unified Spectral Theory-based Dense Subgraph Detection
==================
Fast Spectral Theory-based Algorithms for unified dense subgraphs detection in large graphs.

We propose and formulate the generalized densest subgraph detection problem (GenDS) and fast detection algorithms based on the graph spectral properties and greedy search, i.e., SPECGDS (SpecGreedy) and GEPGDS.

- _Theory & Correspondences_: The unified formulation, GenDS, subsumes many real problems from different applications; and its optimization is guaranteed by the spectral theory.
- _Scalable_: Propose fast and scalable algorithms to solve the unified detection problem over large graphs.
- _Effectiveness_: The performance (solution-quality and speedy) of SpecGreedy is verified on **40** real-world networks; 
                 and it can find some interesting patterns in real applications, like the sudden bursts in research co-authorship relationships    

Environment
=======================
Python 3.6 is supported in the current version.

To install the required libraries, please type
```bash
pip install -r requirements
```
----


Running Demo
========================

Demo for detecting the densest subgraph, please type
```bash
make
```

Datasets Resource
========================

The datasets used are available online; they are from some popular network repositories, including 
[Stanford's SNAP](http://snap.stanford.edu/data), 
[AUS's Social Computing Data Repository](http://socialcomputing.asu.edu/), 
[Network Repository](http://networkrepository.com/),
[Aminer scholar datasets](https://www.aminer.cn/data),
[Koblenz Nwtwork Collection](http://konect.uni-koblenz.de/networks/), and
[MPI-SWS social datasets](http://socialnetworks.mpi-sws.org).
  

Reference
========================
Please acknowledge the following papers if you use this code for any published research.
```
@article{feng2023unified,
  title={Unified Dense Subgraph Detection: Fast Spectral Theory based Algorithms},
  author={Feng, Wenjie and Liu, Shenghua and Koutra, Danai and Cheng, Xueqi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}

@inproceedings{feng2020specgreedy,
  title={SpecGreedy: Unified Dense Subgraph Detection},
  author={Wenjie Feng, Shenghua Liu, Danai Koutra, Huawei Shen, and Xueqi Cheng},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2020},
}
```
