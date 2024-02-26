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

Go to the Mistral folder and run the following commands

```bash
pip install -r requirements.txt
```

After installation, add your sentence to `run.sh` file and run the script `sh run.sh`.

You can also directly run the inference in terminal with code below

```python
python classifier.py --model_name='mistralai/Mistral-7B-Instruct-v0.2' \
                     --sentence="Ferme automatiquement les portes à l'heure prévue."
```
