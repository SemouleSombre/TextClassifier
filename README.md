# Classification with Mistral 7B vs CamemBERT

In this repository, we will perform two classification tasks and compare them.

The first classification will use Mistral 7B Instruct large language model and in the second one, we will finetune Camembert base model on sequence classification.

## Getting started

- **Create a venv**
Create a python virtual environment and install the required dependancies

```bash
python -m venv myenv
```
Activate the virtual environment

```bash
source myenv/bin/activate
```

1. Mistral 7B

```bash
pip install -r Mistral/requirements.txt
```