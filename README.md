jonas-mika/machine-learning
--- 
this repository contains my own *library-style* implementations of the most 
common data preprocessing methods, statistical learning methods and 
performance evaluation methods in both the supervised and unsupervised settings.

the individual ml algorithm implementations are - in their structure - inspired 
from sklearn, as in they are implemented as a class that fundamentally 
contains two main methods `.fit()` and `.predict()`.

all implementations are written in python (requires version python>=3.8.X)
and heavily depend on the following sublibraries:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `pytorch`
- `tqdm`

these dependencies need to be pip installed in order for the implementations to be
run. no guarantee for correctness and maximal efficiency is given.
