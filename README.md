# Pre-Release AdventML
Pre-release of adventml: advanced optimal enzyme temperature prediction. **Models and pipeline will be added soon, keep an eye out!** 

## Pipeline/Update Coming Soon!

Welcome to **AdventML**!

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using Conda Environment with Package Versions](#using-conda-environment-with-package-versions)
  - [Using Conda Environment without Package Versions](#using-conda-environment-without-package-versions)
  - [Using `requirements.txt`](#using-requirementstxt)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Streamlit Web Interface](#streamlit-web-interface)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Accurate Predictions:** Utilizes advanced machine learning models to predict enzyme temperatures.
- **Flexible Input Methods:** Supports uploading FASTA files or entering sequences manually.
- **Multiple Embedding Types:** Choose between `esm1b` and `ProtTransT5XLU50` embeddings.
- **GPU Acceleration:** Option to leverage GPU for faster computations.
- **User-Friendly Interface:** Accessible via command line or an intuitive Streamlit web app.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Conda:** [Download and install Conda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.
- **Git:** To clone the repository. [Download Git](https://git-scm.com/downloads) if needed.
- **CUDA (Optional):** If you plan to use GPU acceleration, ensure CUDA is installed and properly configured.

## Installation

Follow the steps below to set up your AdventML environment and install the necessary dependencies.

### Using Conda Environment with Package Versions

This method ensures reproducibility by installing exact package versions.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/LegallyOverworked/prerelease_adventml.git
   cd prerelease_adventml