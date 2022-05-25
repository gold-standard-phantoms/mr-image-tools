Development
===========

Development of this software project must comply with a few code styling/quality rules and processes:

* Before pushing any code, make sure the CHANGELOG.md is updated as per the instructions in the CHANGELOG.md file. `tox` should also be run to ensure that tests and code-quality checks pass.
* Ensure that a good level of test coverage is kept. The test reports will be committed to the CI system when testing is run, and these will be made available during code review. If you wish to view test coverage locally, run `coverage report`.
* To ensure these code quality rules are kept to, [pre-commit](https://pre-commit.com/) should be installed (see the requirements/dev.txt), and `pre-commit install` run when first cloning this repo. This will install some pre-commit hooks that will ensure any committed code meets the minimum code-quality and is formatted correctly *before* being committed to Git. `mypy`, `Pylint`, `black` and `isort` will be run automatically, and results displayed after a `git commit`. These will also be checked on the GitLab CI system after code is pushed. The tools should also be included in any IDEs/editors used, where possible.
* [mypy](https://github.com/python/mypy) should be run on all source code to find any static typing errors.
* [Pylint](https://pylint.org/) must be used as the linter for all source code. A linting configuration can be found in `.pylintrc`. There should be as few linting errors as possible when checking in code. Code score should be kept high (close to 10), and if linting error are occurring frequently where they aren't expected, relevant changes should be make to the `.pylintrc` file to reflect these (and then code-reviewed).
* Before committing any files, [black](https://black.readthedocs.io/en/stable/) must be run with the default settings in order perform autoformatting on any python files.
* [isort](https://isort.readthedocs.io/en/latest/) should be run on all files before they are committed.
