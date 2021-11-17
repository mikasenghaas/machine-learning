# Educational Machine Learning (EDUML) module for Python
## jonas-mika/edu-machine-learning
---

`edu-machine-learning` (**eduml**) is a python module integrating classical 
machine learning algorithms - covered in the course _Machine Learning_ as 
part of the BSc Data Science at the IT University of Copenhagen.

The module aims to maximise understandability with the goal to be
educationally valuable and aid the teaching in undergraduate level
courses machine learning courses. It's goals is explicitly not to 
maximise ease of maintenance, code reuse or algorithmic performance.

For these purposes, other machine learning libraries and frameworks, 
as noted in the README are recommended.

## Dependencies
---
All implementations are written in Python and depend on a few scientific
and plotting packages from the data science universe of Python. 
These need to be installed in order for the library to work. 
All dependencies can be installed from the `requirements.txt` by running

```
pip install -r requirements.txt
```

Dependencies can also be installed individually. Research the specific
installation procedure on the respective GitHub pages:
- [`numpy`](https://github.com/numpy/numpy)
- [`matplotlib`](https://github.com/matplotlib/matplotlib)
- [`pytorch`](https://github.com/pytorch/pytorch)


## Installation
---
As of now, the only way to obtain the library is by cloning or forking
the repository from this GitHub pages. 
At some point, the module should be published to `PyPi`, s.t. it can
be `pip installed`.


## Idea
---
This module was developed with the idea to provide a programming-based
approach to understanding classical statistical learning methods.

All implementations are divided in sub-modules, that order the different
implemented statistical learning methods by similarity. 
Each machine learning algorithm is designed in a OOP programming approach
and follows a similar style so that different algorithms can easily be
compared. Each class is well-documented through docstrings and code is
commented excessively to maximise understanding.
Each machine learning method is furthermore implemented in a self-contained
way whenever possible, meaning that the code for one implementation depends
on as few code outside its file as possible. Again, this is done to 
maximise understandability of the module.
