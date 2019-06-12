## Example 1

Conditions are:
- [1, 0, 0]
- [0, 1, 0]
- [0, 0, 1]

We give pure noise to the RNN layer.

The target is the condition.

```
TIME_SERIES NOISE ------> RNN conditioned on [1, 0, 0] ------> prediction ------> target is [1, 0, 0]
```

By doing that, we ensure that the condition is actually useful. If the conditioning was to fail, then the RNN could not solve the problem as it only receives noise as input.

Also we can test different noise generators (train on uniform noise and test on gaussian noise).
