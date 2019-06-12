## Example 1

We have one condition that can take the values:
- [1, 0, 0]
- [0, 1, 0]
- [0, 0, 1]

We give pure noise to the RNN layer.

The target is the condition.

```
TIME_SERIES NOISE ------> RNN conditioned on [1, 0, 0] ------> prediction ------> target is [1, 0, 0]
```

By doing that, we ensure that the condition is actually useful. If the conditioning was to fail, then the RNN could not solve the problem as it only received noise as input.

Also we can test different noise generators (train on uniform noise and test on gaussian noise).

## Example 2

We have two conditions:

- [0, 1] and [0, 1, 1] for example.

The target is to count the number of 1s in both conditions. In this case, the target is 3.

Again, we give pure noise to the RNN layer as input.

This example ensures that the network can read from both conditions to make a prediction. If one condition was to be missing, then the task would be impossible to solve.
