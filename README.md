# UCL Machine Reading - FNC-1 Submission

This repository contains the files necessary to reproduce the UCL
Machine Reading group's submission to stage number 1 of the Fake News
Challenge.

Rather than providing seed values and requiring the model to be
retrained, the repository contains the TensorFlow model trained as part
of the submission.

The submission can thus easily be reproduced by loading this model to
make the predictions on the relevant test set.

## Getting started

In order to reproduce the submission, simply download the files in this
repository to a local directory.

### Prerequisites

The model employed was developed, trained and tested using the
following:

```
Python 3.5.2
NumPy 1.11.3
scikit-learn 0.18.1
TensorFlow 0.12.1
```

Please note that compatibility of the saved model with newer versions
of TensorFlow has not been checked. Accordingly, please use the TensorFlow
version listed above.

### Installing

Other than ensuring the dependencies are in place, no separate
installation is required.

Simply execute the `train_pred.py` file once the repository has been
saved locally.

## Reproducing the submission

Execution of the `train_pred.py` file entails the following:

* The train set will be loaded from `train_stances.csv` and
`train_bodies.csv` using the corresponding `FNCData` class defined in
`util.py`.
* The test set will be loaded from `test_stances_unlabeled.csv` and
`train_bodies.csv` using the same `FNCData` class. Please note that
`test_stances_unlabeled.csv` corresponds to the second, amended release
of the file.
* The train and test sets are then respectively processed by the
`pipeline_train` and `pipeline_test` functions defined in `util.py`.
* The TensorFlow model saved in the `model` directory is then loaded
in place of the model definition in `train_pred.py`. The associated
`load_model` can be found in `util.py`.
* The model is then used to predict the labels on the processed test
set.
* The predictions are then saved in a `predictions_test.csv` file in the
top level of the local directory. The corresponding `save_predictions`
function is defined in `util.py`. The predictions made are equivalent to
those submitted during the competition.

Alternatively, as suggested by the organisers of the competition, the
validity of the submission can be checked by training the model with
different seeds and evaluating the average performance of the system.

In order to train the model as opposed to loading the pre-trained one,
carry out the following steps:

* Uncomment the lines of code in `train_pred.py` associated with the
training of the model (lines 86-88 and 91-111)
* Comment the lines of code in `train_pred.py` corresponding to the
loading of the model (lines 115-121)
* Execute `train_pred.py` to train the model and save the consequent
predictions in `predictions_test.csv`. Note that the file name for
the predictions can be changed in section `# Set file names`
at the top of `train_pred.py` if required.

Please note that the predictions saved may still lead to errors with the
official `scorer.py` due to encoding issues, as discussed at length in
the official Slack channels.

## Authors

* **Benjamin Riedel** - Full implementation
* **Sebastian Riedel** - Academic supervision
* **Isabelle Augenstein** - Advice
* **George Spithourakis** - Advice

## License

This project is licensed under the Apache 2.0 License. Please see the
`LICENSE.txt` file for details.

## Acknowledgements

* Richard Davis and Chris Proctor at the Graduate School of Education
of Stanford University for [the description of their development
efforts for FNC-1](https://web.stanford.edu/class/cs224n/reports/2761239.pdf).
The model presented here is loosely based on their
setup.
* Florian Mai at the Christian-Albrechts Universtität zu Kiel for
insightful and constructive discussions during model development.


