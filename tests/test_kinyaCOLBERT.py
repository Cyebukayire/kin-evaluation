from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page

# load the model
model = RAGPretrainedModel.from_pretrained("C4IR-RW/kiny-colbert-free")

# create and index knowledge base
# the first article is about maiz farming, the 2nd about cattle vaccination
knowledge_base = [
  "Ikigori ni igihingwa cy’ingenzi gikura neza mubutaka bukungahaye kandi budatunganyije amazi, ahari izuba rihagije. Abahinzi bakwiye gutera imbuto mu ntiriranywa y’ibihe by’imvura kugira ngo ibihingwa bikiri bito bibone amazi ahagije. Kubagara kenshi no gushyiramo ifumbire y'imborera cyangwa iy'imvugururaburinganire ni ingenzi kugira ngo umusaruro ube mwiza. Iyo amashushu abaye ay'ikigina n'impeke zikaba zishitse, ibigori biba bishobora gusarurwa no guhunikwa.",
  "Gukingira inka ni intambwe y'ingenzi mu kwirinda indwara zica nka Borigo, Uburenge, n'Igifuruta. Kuriha inkingo hakiri kare birahendutse kuruta kuvura amatungo yarwaye. Abahinzi bakwiye gukorana bya hafi n'umuvuzi w'amatungo kugira ngo bagire ingengabihe y'inkingo ihoraho, bityo basigasire ubuzima n'umusaruro w'amatungo yabo."
]

index_path = model.index(index_name="my_index", collection=knowledge_base)

# retrieve data

# question about maize farming
question = "Mfite ikibazo kijyanye n'ubuhinzi bw'ibigori"

RAG = RAGPretrainedModel.from_index(index_path)
results = RAG.search(question)

print(results)

"""
Outputs:

[{'content': "Ikigori ni igihingwa cy’ingenzi gikura neza mubutaka bukungahaye kandi budatunganyije amazi, ahari izuba rihagije. Abahinzi bakwiye gutera imbuto mu ntiriranywa y’ibihe by’imvura kugira ngo ibihingwa bikiri bito bibone amazi ahagije. Kubagara kenshi no gushyiramo ifumbire y'imborera cyangwa iy'imvugururaburinganire ni ingenzi kugira ngo umusaruro ube mwiza. Iyo amashushu abaye ay'ikigina n'impeke zikaba zishitse, ibigori biba bishobora gusarurwa no guhunikwa.",
  'score': 19.476581573486328,
  'rank': 1,
  'document_id': '74d59927-b84f-4140-a35a-17477787453b',
  'passage_id': 0}]
"""


# sbatch --job-name=kinya-test slurm/run.sbatch tests/test_kinya_colbert.py
# to override partition and time: sbatch -p h100day -t 12:00:00 slurm/run.sbatch tests/test_kinya_colbert.py

