#!/bin/bash -v
rsync -avz ../numlabs/* numlabs/.
rsync -avz ../notebooks/lab1/* numeric_notebooks/lab1/.
rsync -avz ../notebooks/lab2/* numeric_notebooks/lab2/.
rsync -avz ../notebooks/lab3/* numeric_notebooks/lab3/.
