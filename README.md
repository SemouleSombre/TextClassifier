from [https://github.com/BirdiD/TextClassifier]

# Classification with Mistral 7B vs CamemBERT

In this repository, we will perform two classification tasks and compare them.

The first classification will use Mistral 7B Instruct large language model and in the second one, we will finetune Camembert base model on sequence classification. We have 90 french sentences that belongs to one of the following classes:
- Intention de recherche d'informations
- Intention d'action
- Intention familière

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

Running the above script will print the predicted category along with some explanation why the category has been chosen in a json format

2. CamemBERT

Here we finetuned a camembert bae model on a small dataset (90 records). 

To train the model, run the following script. Make sure you modifiy the values for your use case:

```python
python train.py --model_name_or_path="camembert/camembert-base" \
                --data_folder_path="Data/Classeur1_catg_phrases.xlsx" \
                --output_dir="output" \
                --hub_model_id="DioulaD/classificateur-intention_camembert" \
                --max_steps="100" \
                --logging_steps="10" \
                --save_steps="20"
```

You can also directly run `sh run_training.sh` in your terminal.

Once training, we can run inference as follows:

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="DioulaD/classificateur-intention_camembert")
pipe("Ouvre la porte et fais vite stp")
```
