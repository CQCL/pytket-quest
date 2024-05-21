# pytket-quest (WIP)

[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

[QuEST](https://quest.qtechtheory.org/) is an open-source high performance simulator of
quantum circuits, state-vectors and density matrices.

## Development

`python -m venv pytket-quest-playground`

`pytket-quest-playground\Scripts\activate.bat`

`pip install -e pytket-quest`

### VS Code

To be able to

    * run/debug unit tests you have to install Testing Explorer extension.
    * use `.pylintrc` you have to install Pylint extension.
    * use `mypy.ini` you have to install Mypy Type Checker extension.
You can find all recommended extensions in `.vscode` folder.

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.
