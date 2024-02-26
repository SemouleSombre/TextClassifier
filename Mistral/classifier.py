from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from kor import JSONEncoder, TypeDescriptor
import pandas as pd
import asyncio
import time
import argparse

from utils import build_llm_pipeline

def split_list(input_list, m):
    """
    Split a list into sublists of size m.
    """
    return [input_list[i:i+m] for i in range(0, len(input_list), m)]

class EncoderAnalyze(JSONEncoder):
    def get_instruction_segment(self) -> str:
        instructions = "Génère l'output sous le format JSON.\n "
        return instructions


class FewShotClassification():
  """
  Class permettant de faire de la classification en utilisant Mistral7B et du few shot learning
  Params:
    list_clause (list): Contient les phrases à classifier. Laisser à None si on veut pas faire la parallélisation
    llm (HuggingFacePipeline LLM)
  """
  def __init__(self, llm, list_clause=None):
    self.promt_description = """[INST] Vous êtes un classificateur d'intention. Vous suivez extrêmement bien les instructions. \n
    Votre objectif est de classifier des phrases correspondant à des requêtes/demandes d'un utilisateur qui vous seront fournies selon les 3 catégories suivantes : 'Intention de recherche d'informations', 'Intention familière' et 'Intention d'actions'. \n
    La requête de l'utilisateur peut-être formulée avec un ton formel, informel ou naturel. Des exemples de requêtes/demandes avec leur classification sont fournis pour vous donner une idée.
    """

    self.list_clause = list_clause
    self.llm = llm

  async def _async_generate(self, chain, clause):
      resp = await chain.arun(text=self._format_clause(clause))
      return resp

  async def generate_concurrently(self):
    """
    Fonction permettant de paralleliser la classification afin d'éviter de la faire phrase par phrase
    """
    s = time.perf_counter()
    instruction_template = self.get_prompt()

    output_schema = self._get_expected_schema()

    chain = create_extraction_chain(self.llm, output_schema, instruction_template=instruction_template, encoder_or_encoder_class=EncoderAnalyze)
    tasks = [self._async_generate(chain, clause) for clause in self.list_clause]
    L = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - s
    return L, elapsed, chain

  def generate(self, clause):
    """
    Fonction permettant de générer du texte sans parallisation
    """
    s = time.perf_counter()
    instruction_template = self.get_prompt()

    output_schema = self._get_expected_schema()

    chain = create_extraction_chain(self.llm, output_schema, instruction_template=instruction_template, encoder_or_encoder_class=EncoderAnalyze)
    L = chain.run(text=self._format_clause(clause))
    elapsed = time.perf_counter() - s
    return L, elapsed, chain

  def _format_clause(self, clause):
    """
    Définir un prompt pour notre classification
    """
    return "Voici la phrase à classifier :"  + "\n" + clause


  def _get_chain(self):
    """
    Définir la chaine (objet langchain) d'extraction pour notre LLM
    """
    instruction_template = self.get_prompt()
    output_schema = self._get_expected_schema()

    chain = create_extraction_chain(self.llm, output_schema, instruction_template=instruction_template, encoder_or_encoder_class=EncoderAnalyze)
    return chain


  def get_prompt(self):
    """
    Formatter les instructions pour la classification
    """
    context = f"Ne pas ajouter d'attributs ou de texte supplémentaires. Classer uniquement la requête de l'utilisateur dans l'une des 3 catégories fournies. \n [/INST]."

    instruction_template = PromptTemplate(
        input_variables=["format_instructions", "type_description"],
        template=(
            f"{self.promt_description}\n\n"
            "{type_description}\n\n"
            "{format_instructions}\n"
            f"{context}\n\n"
        ),
    )
    return instruction_template

  def _get_expected_schema(self):
    """
    On utilise la librairie Kor pour définir nos exemples pour le few shot learning
    """
    schema = Object(
    id="items",
    description="Information concernant la classification de phrases",

    attributes=[
        Text(
            id="categorie",
            description="Correspond au label (la catégorie) à laquelle appartient la requête"
        ),

        Text(
            id="commentaire",
            description="Explication/justification très succinte de classe choisie"
        ),

    ],
    examples=[
        (
            """Pourriez-vous me fournir les dernières directives concernant les procédures de sécurité ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire" : "Il cherche des directives, donc des informations"}],
        ),

        (
            """Tu pourrais me dire si je me tiens mal pour éviter le mal de dos ?""",
            [{"categorie": "Intention familière", "commentaire": "Aucune recherche d'informations. Il s'agit d'une intention purement familière"}],
        ),

        (
            """Engagez le protocole de sécurité pour verrouillage automatique des portes.""",
            [{"categorie": "Intention d'actions", "commentaire" : "Intention d'action car requête relevant d'une action à faire"}],
        ),

        (
            """J'ai besoin de choper les critères de qualité qu'on doit suivre, tu as ça ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire" : "Il cherche les criètes de qualité"}],
        ),
        (
            """Checke si on respire de l'air clean ici.""",
            [{"categorie": "Intention d'actions", "commentaire" : "Intention d'action car requête relevant d'une action à faire"}],
        ),

        (
            """Je requiers une analyse de mon niveau de stress au cours de la journée de travail.""",
            [{"categorie": "Intention familière", "commentaire" : "Intention familière car question sur sa personne"}],
        ),

        (
            """Ferme automatiquement les portes à l'heure prévue.""",
            [{"categorie": "Intention d'actions", "commentaire" : "Intention d'action car requête relevant d'une action à faire"}],
        ),
        
        (
            """Tu as les performances des équipements des derniers mois ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire" : "Cherche les performances"}],
        ),

        (
            """Il me faudrait un retour sur mon stress pendant le boulot.""",
            [{"categorie": "Intention familière", "commentaire" : "Sujet familier concernant l'tulisateur donc intention familière"}],
        ),

        (
            """Quoi de neuf dans le monde des robots ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire":"Ton familier et naturel mais recherhce d'informations"}],
        ),

        (
            """Veuillez activer le système d'alerte en cas de détection de fumée.""",
            [{"categorie": "Intention d'actions", "commentaire" : "Intention d'actions car demande explicite de réalisation d'une action"}],
        ),


        (
            """Pouvez-vous me rappeler de faire des pauses régulières pour éviter la surcharge cognitive ? """,
            [{"categorie": "Intention familière", "commentaire" : "Question familière pour faire des pauses"}],
        ),

        (
            """Des idées pour me rappeler de boire suffisamment ?""",
            [{"categorie": "Intention familière", "commentaire":"Il veut être rappelé d'un sujet le concernant directement donc intention familière"}],
        ),

        (
            """Fais tourner le check-up des machines pour voir si tout roule.""",
            [{"categorie": "Intention d'actions", "commentaire" : "Demande explicite de vérification donc intention d'action"}],
        ),

        (
            """Comment on fait pour la maintenance des robots, t'as un tuto ou un truc du genre ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire":"car cherche à savoir comment on fait la maintenance"}],
        ),

        (
            """Quelles sont les recommandations pour l'optimisation des flux de travail ?""",
            [{"categorie": "Intention de recherche d'information", "commentaire":"Il veut avoir des info sur les recommandations"}],
        ),

        ]
    )
    return schema



if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run few shot classification")

    parser.add_argument("--model_name", type=str, required=True, help="HF model name")
    parser.add_argument("--sentence", type=str, required=True, help="Sentence to be classified in one of the category")
    parser.add_argument("--temperature", type=float, default=0.001, help="Temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--max_new_tokens", type=int, default=2080)

    # Parse the command-line arguments
    args = parser.parse_args()

    llm = build_llm_pipeline(args.model_name, args.max_new_tokens, args.temperature, args.repetition_penalty)

    classifier = FewShotClassification(llm)
    output, elapsed, chain = classifier.generate(args.sentence)

    print(f"The output is : {output} \n")
    print("-----------------------")
    print(f"The elapsed time is : {elapsed} \n")

    print("-----------------------")
    print(f"""This is what the LLM sees : {chain.prompt.format_prompt(text=args.sentence).to_string()} \n""")


