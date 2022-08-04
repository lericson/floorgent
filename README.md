# FloorGenT

Floor plans are the basis of reasoning in and communicating about indoor
environments. In this paper, we show that by modelling floor plans as sequences
of line segments seen from a particular point of view, recent advances in
autoregressive sequence modelling can be leveraged to model and predict floor
plans. The line segments are canonicalized and translated to sequence of tokens
and an attention-based neural network is used to fit a one-step distribution
over next tokens. We fit the network to sequences derived from a set of
large-scale floor plans, and demonstrate the capabilities of the model in four
scenarios: novel floor plan generation, completion of partially observed floor
plans, generation of floor plans from simulated sensor data, and finally, the
applicability of a floor plan model in predicting the shortest distance with
partial knowledge of the environment.

Visit the [project website](https://lericson.se/floorgent/).

## Dependencies, requirements files

FloorGenT was developed on a Python 3.6 stack. We recommend using `pyenv` or
similar to install a Python 3.6.x interpreter, then create a virtualenv.

`requirements.txt` are the high-level package requirements, you can choose to
install this by `./env/bin/pip install -r ./requirements.txt` and allow pip to
decide what subdependencies to install; or you can choose to use the file
`./requirements-snapshot.txt` which is a full snapshot of the versions we used.

