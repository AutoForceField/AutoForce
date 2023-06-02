# Developer Guide



## Contributing

For contributing to the source code, clone the repository and create a new branch:
```
git clone https://github.com/AutoForceField/AutoForce.git
git checkout -b my-branch
```
Then proceed with your contributions while following the rules and guidlines which
are discussed in the following secions for maintaining the code quality.

After making sure that all standards are met, the branch can be pushed for review
and pull request:
```
git push --set-upstream origin my-branch
```

TODO: pull requests.



## Branching

A branch should implement a single feature and then immediately be merged to the
main branch.
If a branch diverges too much from the main branch, it may never be merged.
Exceptions are the adding new features through new modules and subpackages with
minimal changes of the existing files.



## Design Patterns

- Abstract Base Classes (ABCs): TODO
- Functional mindset: TODO
- SOLID principles: TODO



## Versions

TODO:



## Coding Style

The coding style and checks are automated using `pre-commit` which can be installed by:
```sh
conda install pre-commit
```
Then the following command can be used to make sure that the proposed commit conforms
with the coding style:
```sh
pre-commit run --all
```
Note that a "pull request" is only merged to the main branch only if all the checks pass.



## Type Checking

The code for AutoForce should be fully typed.
Automatic type checking is handled by `mypy` as a part of `pre-commit`.



## Testing

All the functions and classes should have rigorous tests in suitable files.

TODO: where should tests be written?



## Imports

TODO:
