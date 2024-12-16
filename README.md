# RAMP starting-kit object detection in Laser Wakefield Acceleration Simulation outputs from the Smilei PIC code 

Authors : Mathieu Lobet, Hiba Taher, Martial Mancip, Merieme Bourenane (Maison de la Simulation, Saclay), Francesco Massimo (LPGP, Univ. Paris-Saclay), François Caud, Thomas Moreau (DATAIA, Univ. Paris-Saclay)

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [dedicated notebook](lwfa_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```
In case of a very long ramp-test, you can select a subset of the data to run it with
the '--quick-test' option (if available in problem.py)
```bash
ramp-test --submission my_submission --quick-test
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
