# Student Performance Indicator


## First install the dependencies

```bash
pip install -r requirements.txt
```
or
```bash
pip install -e .
```

## Steps followed to create this
1. Create **requirements.txt** and in it include `-e .` also so that the package is installed in editable mode when we run pip install -r requirements.txt

2. Create a **setup.py** file with the necessary information about the package, such as name, version, description, author, and the packages to include.

3. Create the **src directory** and add `__init__.py` file to make it a package.
    **Note :**  Remember to include `__init__.py` in all subdirectories that you want to be treated as packages.

4. Created the basic structure including components and pipeline folder.

5. Exception handling and logging is added in the code.
