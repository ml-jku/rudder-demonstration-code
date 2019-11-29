# rudder-demonstration-code
Code for demonstration example-task in RUDDER blog. For more materials see the [RUDDER repo](https://github.com/ml-jku/rudder) or the [practical step-by-step guide to applying RUDDER in PyTorch](https://github.com/widmi/rudder-a-practical-tutorial).

## A simple RUDDER demo:

We provide an implementation of 
the introductory example-task of repairing pocket watches 
to demonstrate the efficiency of RUDDER.
In this delayed reward task both TD and MC methods have problems.
In contrast, RUDDER is efficient since it pushes 
the expected future rewards close to zero.
TD learning is impeded since it has to 
correct the biases of its estimates. 
MC learning is impeded since it has problems with high variance.

#### Requirements:
* python3 >= 3.6
* pytorch >= 1.0.1
* numpy
* argparse
* if plots are desired: matplotlib >= 3.1.0

#### Running the demonstration: 
```
python3 watch_repair.py --policy_learning=RUDDER
```

Please use the `policy_learning` argument to change the policy learning method.
Valid options are: 
* `RUDDER`: RUDDER Q-value estimation as described in the main paper. (completes after approx.: 140-250sec | 1,900-3,800episodes)
* `TD`: for Q-Learning (completes after approx.: 400-470sec | 71,800-71,900episodes)
* `MC`: for Monte-Carlo control. (completes after approx.: 150->600sec | 27,700->240,000episodes)

Approximate runtime and number of episodes were estimated from random seeds {1, 2, 3}, where MC did not complete within 10 minutes for 2 random seeds.

#### Problem Description:
##### Environment:
In this task you have to repair pocket watches and then sell them.
For a particular brand of watch you have to decide
whether repairing pays off. 
There are 4 uniformly randomly chosen initial states 
which indicate the brand of watch. 
The possible actions are repairing (a=0) or not repairing (a=1).
After this first decision (repair or not repair),
the next transitions only depend on the state-transition probabilities.

Repairing a particular brand pays off if
the sales price
minus the expected immediate repair costs
minus the expected future delivery costs is positive.

Repairing the watch (a=0) has an immediate negative reward (repair costs).
The immediate repair costs are normal random variables:
* repair cost for brand0: (mean 1, var 2.0) 
* repair cost for brand1: (mean 4, var 2.0)
* repair cost for brand2: (mean 5, var 2.0)
* repair cost for brand3: (mean 24.5, var 1.0)


The sales price is known (deterministic value):
* brand0: 18
* brand1: 28
* brand2: 31
* brand3: 59


Delivery costs are unknown.

##### Task:

You have to estimate
the expected brand-related delivery costs,
which are e.g. packing costs.
These brand-related costs are normal random variables:
* brand0: (mean 0.5, var 1.5)
* brand1: (mean 2.5, var 1.5)
* brand2: (mean 4.5, var 1.5)
* brand3: (mean 18, var 1.5)

These brand-related costs are superimposed by 
brand-independent general delivery costs 
for shipment (e.g. time spent for delivery).
General delivery costs
are indicated by patterns in the input, e.g. traffic jams or flat tires
which delay delivery.
These events are the same for each brand.
Each event has a cost and a probability to happen at any time step
* traffic jams:  costs 0.1, with probability of 0.1
* flat tires: costs 7.0, with probability of 0.05

Every episode is 50 time steps long. 
The average general delivery cost is 18.
Average general delivery costs: 18.0 = 50 * ( 0.1 * 0.1 + 7 * 0.05 )


For example, repairing brand0 does not pay off, i.e. repairing it is a **poor decision**:
18 - 1 - 0.5 - 18.0 = -1.5

But, repairing brand1 does pay off:
28 - 5 - 2.5 - 18.0 = 2.5

The state is coded using 4 features:
* Brand of watch
* Status of watch (repaired or not)
* Number of traffic jams which increase delivery costs
* Number of flat tires which increase delivery costs
* Time
