Notes from tweaking the model a bit to see if I can get better performance:

- Seems like as I increase the epochs the loss value decreases linearly
- Likewise, increasing the number of neurons in each layer seems to have a similar affect. However, once I increased the neuron count to 64 in the first layer and 32 in the second layer, loss decreases seemed to diminish
- In all of these cases, other than the extra epochs iterated through, the performance of the system did not seem to be affected
- Seems like performance increases the most when adding neurons to both layers of the model and when each layer has the same number of neurons
- When adding a fourth layer, it seemed like there was an initial boost when using a smaller number of epochs but saw diminishing returns when adding more epochs.

Overall, it seems like adding more neurons to a limited number of layers while using an increasing amount of epochs is the best way to train this model
