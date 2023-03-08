# Contributing to MedDoCan

## Developing locally

install git, clone the repository and configure correctly git with your user name and email

```console
$ git clone https://github.com/GuiGel/MedDocAn.git
Cloning into 'MedDocAn'...
```

cd into the *MedDocAn* folder

```console
$ cd MedDocAn
```

Then configure your git username ...

```
$ git user.name XXXXXX
$ git user.email XXXXX
```

The next step is to install the project as well as all the required dependencies. For that we use **pyenv** to deals with various python version and **Poetry** to manage the dependencies versions.

be sure to use the right python 3.8 or more.

```console
$ poetry config virtualenvs.in-project true
$ poetry install
```
