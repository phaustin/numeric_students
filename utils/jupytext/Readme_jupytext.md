# jupytext howto

* https://github.com/mwouts/jupytext

  `conda install  -c conda-forge jupytext`

* to move from a notebook to a py file:

   jupytext --to py:percent *ipynb

* to move from a py file to a notebook

  jupytext --to notebook *py  --sync --execute

* to copy change in a py file automatically to the ipynb
  partner while editing in jupyter:

  - generate a config file if you don't have one:

    `jupyter notebook --generate-config`

  - overwrite or extract the uncommented lines from

    [jupyter_notebook_config.py](https://github.com/phaustin/numeric/blob/master/utils/jupytext/jupyter_notebook_config.py)

    and restart jupyter server.  After this, edit py files only, treat ipynb
    files as output.



  
