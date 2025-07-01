# Python Package Template

## Quick Start

1. (Optional) Click the "Use this template" button on GitHub to create a new repository based on this template.

2. Clone this repository:

    ```bash
    git clone https://github.com/Bardreamaster/python_package_template.git
    ```

3. Edit permissions to allow execution of the `init_package.sh` script:

    ```bash
    sudo chmod +x init_package.sh
    ```

4. Initialize the package with the desired name, author, and email:

    ```bash
    ./init_package.sh --name your-package-name --author "Your Name" --email "your.email@example.com"
    ```

5. After running the script, you can delete the `init_package.sh` script.
6. (Optional) Config GitHub Actions if you want to use it for CI/CD. The workflow is already set up in the `.github/workflows` directory.

    - `gh-pages.yml` for deploying documentation to GitHub Pages.
    - `pypi.yml` for publishing the package to PyPI.
    - `tests.yml` for running tests.

    To configure GitHub Actions, you need to:

    1. Edit the workflow rules under section `on:` in the YAML files to specify when the workflows should run.
    2. [Create environment](https://docs.github.com/en/actions/how-tos/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment#creating-an-environment) named `pypi` and `github-pages` for your repository in GitHub settings.
    3. If you want to publish the package to PyPI, you need to [configuring PyPI’s trusted publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/#configuring-trusted-publishing)
    4. If you want to deploy documentation to GitHub Pages, you need to [enable GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) for your repository and set the source branch to `gh-pages`.
    5. If you want to use Codecov for code coverage, you need to setup it by [following the quick start guide](https://docs.codecov.com/docs/quick-start).

7. (Optional) Update license information in the `LICENSE` file. The default license is MIT, but you can change it to any other license you prefer.
8. 🎉 Well Done! You can now start developing your Python package.

    - Add more features and functionality to your package.
    - Write tests to ensure your code works as expected.
    - Update the documentation.
    - ......

## Directory Structure

```plaintext
python_package_template/
├── .devcontainer/          # Development container configuration
├── docs/                   # Documentation files
├── src/
    ├──python_package_template/  # Source code for the package
├── tests/                  # test files
├── .gitignore              # Git ignore file with common python patterns
├── .pre-commit-config.yaml # Pre-commit configuration file
├── init_package.sh         # Script to initialize the package
├── LICENSE                 # License file
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file, the README
```

### devcontainer

This repository includes a `.devcontainer` folder for setting up a development container using Visual Studio Code. To use it, follow these steps:
1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in Visual Studio Code.
2. Open the repository in Visual Studio Code.
3. Press `F1` and select "Remote-Containers: Reopen in Container" from the command palette.
4. Wait for the container to build and start. This may take a few minutes, especially the first time.
5. Once the container is running, you can start developing your Python package in a consistent environment.

## Related Sources

- [Python Packaging User Guide](https://packaging.python.org)
