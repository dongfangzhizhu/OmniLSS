#!/bin/bash

echo "===== Python ====="
python3 --version

echo "===== R ====="
R --version

echo "===== Installing renv ====="

R -e "if (!require('renv')) install.packages('renv', repos='https://cloud.r-project.org')"

echo "===== Restoring renv ====="

R -e "renv::restore(prompt = FALSE)"

echo "===== Environment Ready ====="