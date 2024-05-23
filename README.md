SpecGreedy: Unified Dense Subgraph Detection
==================
**SpecGreedy** is a unified fast algorithm for the generalized densest subgraph detection problem (GenDS)
based on the graph spectral properties and a greedy peeling approach. 

- _Theory & Correspondences_: the unified formulation, GenDS, subsumes many real problems from different applications; 
                            and its optimization is guaranteed by the spectral theory.
- _Scalable_: SpecGreedy runs linearly with the graph size.
- _Effectiveness_: The performance (solution-quality and speedy) of SpecGreedy is verified on **40** real-world networks; 
                 and it can find some interesting patterns in real applications, 
                 like the sudden bursts in research co-authorship relationships    


**[The repo will be updated soon]**

Datasets
========================

The datasets used are available online; they are from some popular network repositories, including 
[Stanford's SNAP](http://snap.stanford.edu/data), 
[AUS's Social Computing Data Repository](http://socialcomputing.asu.edu/), 
[Network Repository](http://networkrepository.com/),
[Aminer scholar datasets](https://www.aminer.cn/data),
[Koblenz Nwtwork Collection](http://konect.uni-koblenz.de/networks/), and
[MPI-SWS social datasets](http://socialnetworks.mpi-sws.org).
  

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


Reference
========================
If you use this code as part of any published research, please acknowledge the following papers.
```
@inproceedings{feng2020specgreedy,
  title={SpecGreedy: Unified Dense Subgraph Detection},
  author={Wenjie Feng, Shenghua Liu, Danai Koutra, Huawei Shen, and Xueqi Cheng},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2020},
}
```
