# TODO
## Educational Machine Learning (EDUML) module for Python
---

### General
---
- [ ] Write Exception classes
- [ ] Further develop validation function to use in implementations
- [ ] Publish package on `pip`

### eduml.linear_model
---
- [ ] Improve models default parameters for all GD based algorithms -> convergence criteria

### eduml.metrics
---
- [ ] Implement recall, precision, confusion_matrix, ...
- [ ] Implement R2-Score, and more regression params

### eduml.plotting
---
- [ ] Implement heat map for confusion matrix

### eduml.utils
---
- [ ] 

### eduml.svm
---
- [ ] Implement SoftMarginClassifier (by changing the objective and constraining functions in the dual optimisation)
- [ ] Implement SVM (based on the implementation of the SoftMarginClassifier) by introducing kernels

### eduml.tree
---
- [ ] Implement ExtraTreeClassifier
- [ ] Implement Feature Importance Parameter (weighted average, where each node's weight is equal to the number of 
      samples it splits, normalised in the end, st. sum of feature importances is 1)

### eduml.ensemble
---
- [ ] Implement RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, BaggingRegressor
- [ ] Look into how to speed up ensembles through parallelisation
- [ ] Implement Boosting Algorithms: AdaBoost and 

