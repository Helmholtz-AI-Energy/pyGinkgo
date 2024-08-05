## Contributing to pyGinkgo

Thank you for your interest in contributing to pyGinkgo. To get started, follow the following instructions:

 * If you want to report a bug, or propose a new feature, you can file an [Issue](https://github.com/Helmholtz-AI-Energy/pyGinkgo/issues/new).
 * To set up your environment for development, follow the instructions in [README.md](README.md).
 * We strongly recommend getting in touch with the core developers, by filing and/or commenting on an Issue before starting to work on a contribution.
 * Rebasing and pushing the changes:

 1. `git add`, `pre-commit run --all-files` and `git commit` as needed;
 2. `git rebase -i main` to rebase and tidy up your commits;
 3. `git push` to publish to the remote repository.

*The following is based on the SciPy community's [Contributing to NumPy](https://numpy.org/doc/stable/dev/) guidelines.*

#### Getting the Source Code

* Go to [https://github.com/Helmholtz-AI-Energy/pyGinkgo/](https://github.com/Helmholtz-AI-Energy/pyGinkgo/) and click on the “fork” button to create your own copy of the project.

* Clone the repository to your computer by running:

```
git clone git clone https://github.com/<YOUR-USERNAME>/pyGinkgo.git
```

* Change your working directory to the cloned repository:

```
cd pyGinkgo
```

* Add the original pyGinkgo repository as your upstream:

```
git remote add upstream https://github.com/Helmholtz-AI-Energy/pyGinkgo.git
```

* Now, `git remote -v` will show two remote repositories named:
    * `upstream`, which refers to the main pyGinkgo repository
    * `origin`, which refers to your personal fork of pyGinkgo

#### Developing Contributions

* Pull the latest changes from upstream:

```
git checkout main
git pull upstream main
```

* Enforcing coding conventions:

In order to use this framework, please install the pre-commit hook with

```
pre-commit install
````

* Commit locally as you progress:

```
git add
pre-commit run --all-files
git commit
```

Use a properly formatted commit message, write tests that fail before your change and pass afterwards.

#### Publishing your Contributions

* Before publishing your changes, you might want to rebase to the main branch and tidy up your list of commits, keeping only the most relevant ones and "fixing up" the others. This is done with interactive rebase or `git rebase -i`. Here's an excellent [tutorial](https://www.atlassian.com/git/tutorials/merging-vs-rebasing). This should only be done **before** pushing anything to the remote repository!

* Push your changes back to your fork on GitHub:

```
git push origin features/123-boolean-operators
```

* Enter your GitHub username and password (advanced users can remove this step by connecting to GitHub with SSH).

* Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self-explanatory. Then click the button to submit it.

* If your commit introduces a new feature or changes functionality, **please explain your changes and the thinking behind them**. This greatly simplifies the review process. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for a review.

* Phrase the PR title as a changelog message and make sure the PR is properly tagged ('enhancement', 'bug', 'ci/cd', 'chore', 'documentation').

#### Review Process

* Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation, and style. Every single developer working on the project has their code reviewed, and we’ve come to see it as a friendly conversation from which we all learn and the overall code quality benefits. Therefore, please don’t let the review discourage you from contributing: its only aim is to improve the quality of the project, not to criticize (we are, after all, very grateful for the time you’re donating!).

* To update your PR, make your changes on your local repository, commit, run tests, and push to your fork. As soon as those changes are pushed up (to the same branch as before) the PR will update automatically. If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a PR comment.

* Various continuous integration (CI) services are triggered after each PR update to build the code, run unit tests, measure code coverage, and check the coding style of your branch. The CI tests must pass before your PR can be merged. If CI fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. To avoid overuse and waste of this resource, test your work locally before committing.

* There might also be a "failed" red cross, if the test coverage (i.e. the test code lines) is not high enough. There might be good reasons for this that should be properly described in the PR message. In most cases however, a sufficient test coverage should be achieved through adequate unit tests.

* A PR must be approved by at least one core team member before merging. Approval means the core team member has carefully reviewed the changes, and the PR is ready for merging.

* If the PR relates to any issues, you can add the text `#<ISSUE-NUMBER>` to insert a link to the original issue and/or another PR. Please do so for all relevant topics known to you.

#### Document Changes

* Make sure to reflect changes in the code in the functions docstring and possible description in the general documentation.

* If your change introduces a deprecation, make sure to discuss this first on GitHub and what the appropriate deprecation strategy is.

#### Divergence between upstream/main and your feature branch

If GitHub indicates that the branch of your PR can no longer be merged automatically, you have to incorporate changes that have been made since you started into your branch. Our recommended way to do this is to rebase on `main`.

## Guidelines

* All code should have tests (see test coverage below for more details).

* No changes are ever merged without review and approval by a core team member. Feel free to ping us on the PR if you get no response to your pull request within a week.


## Test Coverage

* Pull requests (PRs) that modify code should either have new tests, or modify existing tests to fail before the PR and pass afterwards. You should run the tests before pushing a PR.

* Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%.
