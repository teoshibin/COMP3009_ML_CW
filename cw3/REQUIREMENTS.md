1. Download [Anaconda](https://www.anaconda.com/distribution/), a Python package and environment manager. Download and install Python 3.7 version binary from this page

2. After installing the Conda binary, open ‘Anaconda Prompt’ from the Windows Start Menu and execute the following commands:

3. Create new Conda virtual environment called 'mle_tf' that uses Python 3.7:
    ```batch
    conda create -n mle_tf python=3.7 
    ```

4. Enter above created virtual environment:
    ```batch
    conda activate mle_tf
    ```

5. Install explicitly specified versions of packages
    ```batch
    conda install -c conda-forge numpy=1.17.2
    conda install -c conda-forge tensorflow=1.13.1
    conda install -c conda-forge matplotlib=3.1.1
    conda install -c conda-forge scipy=1.3.1
    ```
    or one liner for window batch
    ```batch
    conda install -y -c conda-forge numpy=1.17.2 & conda install -y -c conda-forge tensorflow=1.13.1 & conda install -y -c conda-forge matplotlib=3.1.1 & conda install -y -c conda-forge scipy=1.3.1
    ```