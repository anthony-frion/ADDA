.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codebase.helmholtz.cloud/m-dml/da-tools/badges/main/pipeline.svg
    :target: https://codebase.helmholtz.cloud/m-dml/da-tools/pipelines/commits/main

.. image:: https://codebase.helmholtz.cloud/m-dml/da-tools/badges/main/coverage.svg
    :target: https://codebase.helmholtz.cloud/m-dml/da-tools/pipelines/commits/main

===============================
Documentation of the da-tools
===============================

da_tools is a set of differentiable implementations of common data assimilation algorithms written in pytorch.

Full documentation at https://m-dml.pages.hzdr.de/da-tools/

--------------------------------------------

Installation
------------
To install you have to clone the repository and install it with pip:

.. code-block:: bash

   git clone https://codebase.helmholtz.cloud/m-dml/da-tools.git
   cd da-tools
   pip install -e .

--------------------------------------------

Building this documentation
---------------------------
To build this documentation you have to install the requirements in the `docs_environment.yaml` or
`dev_environment.yaml`. Then from within the `docs` folder you can build the documentation with:

.. code-block:: bash

   make html

The created documentation can opened in a browser using the `docs/build/html/index.html` file.


--------------------------------------------


Contributing
------------

1. Create an issue describing what you want to change/add/remove ...
2. Create a Merge-Request from that Issue (blue button in the upper right). With each merge-request a branch is automatically created. Also the MR will be automatically created with the "DRAFT" keyword, which means it can not be merged directly.
3. On your machine switch to that newly created branch of that MR.

    .. code-block:: bash

        git switch branch-name

4. Implement your changes. You can push as many commits as you want to this branch.
5. As soon as you are done with all changes make sure that all existing unit-tests pass and if you implemented something new, make sure it also got a unit-tests
6. run pre-commit (see below)
7. push one more time, if the pre-commit run changed your code.
8. On GitLab go to your Merge request and click "Edit" in the top right corner. Remove the tick at "Mark as draft" and save changes.
9. Tell a maintainer that your MR is ready for being reviewed.
10. If the reviewer is happy he or she will merge your code to the main branch. Else you might have to make a few adjustments.

Pre-Commit
-----------

Please install the conda environment in `dev_environment.yaml`. Before you commit, please run

.. code-block:: bash

    pre-commit run --all-files


from the main directory to make sure that your code is formatted correctly.
We are using `black` for formatting and `flake8` for linting.
