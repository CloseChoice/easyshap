# easyshap

## Installation
1. Optional: create a virtualenv
### Linux
```bash
python -m venv .venv
source .venv/bin/activate
```
### Windows
```bash
python -m venv .venv
source .venv/Scripts/activate
```
2. Install the package
Navigate in the root folder of this package. Then execute the following command:
```bash
pip install easyshap[plots,test,test-tf]
```
Any additional dependencies in the square brackets are optional.

### Google colab

```bash
pat = "<GITHUB_PERSONAL_ACCESS_TOKEN>"
!git clone --branch feature/eshap-compare https://{pat}@github.com/CloseChoice/easyshap.git
pip install /content/easyshap[plots,test,test-tf]
```
   
