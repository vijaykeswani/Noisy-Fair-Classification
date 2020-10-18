This repository contains code for an algrorithm to generate a fair classifier when the protected attribute is noisy.

The paper for this algorithm is the following:

*Fair Classification with Noisy Protected Attributes: A Framework with Provable Guarantees* (https://arxiv.org/abs/2006.04778)

The current implementation supports flipping noises in the protected attributes and two types of fairness metrics:
* Statistical parity
* False Discovery

The algorithms can be run on the pre-processed datasets provided - Adult and Compas. 
Since we use AIF360 repository to load the pre-processed datasets directly, we also include a clone of their repository.

To run the algorithm, the following parameters are required:
* Protected attribute ("sex" or "race")
* Noise parameters (eta0 and eta1 in the paper)
* Desired fairness level tau
* Number of repetitions

        $ python3 noisy_fairlearn.py race 0.3 0.1 0.9 10

For every repetition, this will generate a noisy dataset using (eta0, eta1) flipping noises and then construct fair classifiers (with respect to statistical and false positive metrics separately) for input fairness parameter tau.


