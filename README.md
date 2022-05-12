# RankedCUCB: Ranked Prioritization of Groups in Combinatorial Bandit Allocation

Lily Xu, 2022

This code implements and evaluates algorithms for the paper ["Ranked Prioritization of Groups in Combinatorial Bandit Allocation"](https://arxiv.org/abs/2205.05659) from AAMAS-22. In this paper we introduce RankedCUCB, a multi-armed bandit approach to select combinatorial actions that optimize the prioritization-based objective introduced. This code also implements and compares against a variety of baselines.

```
@inproceedings{xu2022ranked,
  title={Ranked Prioritization of Groups in Combinatorial Bandit Allocation},
  author={Xu, Lily and Biswas, Arpita and Fang, Fei and Tambe, Milind},
  booktitle={Proc.~31st International Joint Conference on Artificial Intelligence (IJCAI-22)},
  year={2022},
}
```

This project is licensed under the terms of the MIT license.


## Usage

To execute the code and run experiments comparing RankedCUCB and the baselines, run:
```sh
python driver.py
```

To vary the settings, use the options:
```sh
python driver.py -N 25 -B 5 -G 5 -T 500 -l 0.4 -s 0 -V
```

The options are
- `N` - number of arms (e.g., targets, ports, or neighborhoods in the wildlife, COVID testing, or mobile health domains)
- `B` - budget (total number of resources to spend)
- `G` - number of demographic groups (e.g., wildlife species or human age group)
- `D` - domain {synthetic, wildlife}
- `T` - horizon
- `l` - lambda
- `s` - random seed
- `V` - whether to use verbose output (default False)

## Files

- `driver.py` - main script to execute one set of experiments
- `adversary.py` - set up reward functions ([adversary.py](https://github.com/lily-x/dual-mandate/blob/master/adversary.py) from [AAAI-21 paper on Dual-Mandate Patrols](https://arxiv.org/abs/2009.06560))
- `ranked_cucb.py` - abstract class for RankedCUCB objective + algorithm
- `ranked_linear.py` - instantiated class for the linear objective + algoroithm
- `get_domain.py` - sets up the two experimental domains (wildlife and synthetic). Note that we are
- `process_results.py` - after running full set of experiments with different seeds, this script will consolidate the CSVs from each domain, computing mean and standard error, to be used in the paper

## Requirements
- python==3.7.4
- numpy==1.17.2
- pandas==0.25.1
- matplotlib==3.1.1
- scipy==1.3.1
- pytorch==1.4.0
- scikit-learn==0.21.3
- gurobi==9.1.2
