# Contributor Guide

Thank you for considering contributing to our project! This guide outlines the process for setting up your development environment, coding standards, and submitting contributions.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [Code Style and Standards](#code-style-and-standards)
4. [Running Tests](#running-tests)
5. [Documentation](#documentation)
6. [Submitting Contributions](#submitting-contributions)
7. [License](#license)

## Getting Started

Before you begin contributing, please take a moment to review the following:

- **Familiarize yourself with the project:** Explore the project’s documentation, codebase, and issues to understand the current state of the project.
- **Search for existing issues:** Check the issue tracker to ensure your contribution doesn't duplicate existing work.
- **Join the discussion:** If you're unsure about anything, feel free to open an issue or start a discussion.

## Setting Up the Development Environment

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<GIT_REPO_OWNER>/<GIT_REPO_NAME>.git
    cd <GIT_REPO_NAME>
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -e .[test,doc]
    ```

4. **Set up pre-commit hooks:**
    ```bash
    pre-commit install
    ```

## Code Style and Standards

- **Follow PEP 8:** Adhere to Python's PEP 8 style guide.
- **Use type hints:** Wherever possible, use type hints to improve code clarity.

## Running Tests

1. **Run the test suite:**
    ```bash
    pytest
    ```

2. **Check for linting errors:**
    ```bash
    pre-commit run --all-files
    ```

## Documentation

Ensure that any new functionality or changes are well-documented:

- **Docstrings:** Add docstrings to all new functions and classes.
- **AutoDocumentation:** Documention is autogenerated when pushed to main branch using pdoc and github-pages. To preview the documentation locally for a branch, checkout the branch, and run `scripts/pdoc.sh`. To build one time you can run `scripts/build-documentation` but to see live updates run `scripts/pdoc.sh --port PORT` and `--no-browser` if applicable

## Submitting Contributions

Only branches that follow the expression `feature/*` will be submitted to github actions.

1. **Create a new branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```

2. **Commit your changes:**
    ```bash
    git commit -m "Add new feature X"
    ```

3. **Push your branch:**
    ```bash
    git push origin feature/your-feature-name
    ```

4. **Open a Pull Request (PR):** Submit your PR via GitHub and link it to relevant issues.
"""
