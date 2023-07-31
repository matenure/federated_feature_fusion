This file includes information of `metr-la.npz`. For `pems-bay.npz`, it is similar. The corresponding graph information is given in a separate pickle file. The file name identifies the data set.

Prepared by Jie Chen.

Load the file in python in the following way:

```python
>>> import numpy as np
>>> dat = np.load('metr-la.npz')
>>> dat['x'].shape
(2856, 207, 12, 1)
>>> dat['y'].shape
(2856,)
>>>
>>> import pickle
>>> f = open('adj_mx_la.pkl','rb')
>>> pdata = pickle.load(f, encoding='latin1')
>>> pdata[2]
array([[1.       , 0.       , 0.       , ..., 0.       , 0.       ,
        0.       ],
       [0.       , 1.       , 0.3909554, ..., 0.       , 0.       ,
        0.       ],
       [0.       , 0.7174379, 1.       , ..., 0.       , 0.       ,
        0.       ],
       ...,
       [0.       , 0.       , 0.       , ..., 1.       , 0.       ,
        0.       ],
       [0.       , 0.       , 0.       , ..., 0.       , 1.       ,
        0.       ],
       [0.       , 0.       , 0.       , ..., 0.       , 0.       ,
        1.       ]], dtype=float32)
>>> f.close()
```

`dat['x']` contains all inputs. It has not been split for train/val/test yet.

- 2856: number of examples
- 207:  number of local parties/sensors/data sources
- 12:   time series length
- 1:    number of variates for each time series

`dat['y']` contains all targets corresponding to the inputs.

`pdata[2]` contains the graph adjacency matrix.