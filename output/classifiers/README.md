This folder contains different training rounds where the theta bands were modified.
During training, it is necessary to define the bands for extracting features like so:

```
custom_bands =  [
          (0.4, 1, 'sdelta'), 
          (1, 4, 'fdelta'), 
          (4, 8, 'theta'), <- This was modified in the different runs
          (8, 12, 'alpha'), 
          (12, 16, 'sigma'), 
          (16, 30, 'beta')
      ]
```

In general, all rounds train and predict with very similar performance. Folders are named using the `theta_low_high` pattern. For example, `theta_5_8` contains the classifiers run with `(5, 8, 'theta')`


The classifiers given under no folder are the same as `theta_4_8`