from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB 
import pandas as pd
import medcat
import numpy as np

# Create and load the CDB (Concept Database)
cdb = CDB()
cdb.load_dict(u"cdb-medmen.dat")

# Create and load the Vocabulary
vocab = Vocab()
vocab.load_dict(u"vocab.dat")

# Create CAT - the main class from medcat used for concept annotation
cat = CAT(cdb=cdb, vocab=vocab)

# Set a couple of parameters, they are usually set via environments, but
#here we will do it explicitly. You can read more about each option in the 
#medcat repository: https://github.com/CogStack/MedCAT
cat.spacy_cat.PREFER_FREQUENT = True
cat.spacy_cat.PREFER_ICD10 = False
cat.spacy_cat.WEIGHTED_AVG = True
cat.spacy_cat.MIN_CONCEPT_LENGTH = 3 # Ignore concepts (diseases) <= 3 characters
cat.spacy_cat.MIN_ACC = 0.2 # Confidence cut-off, everything bellow will not be displayed 

# text = """former smoker"" and ""never smoker"""
# text = "Says no ETOH use, and says occasional alcohol use"
datasetInPandasDatFrame = pd.read_csv("Logistic_Notes.csv")

# datasetInPandasDatFrame = pd.read_csv("Epic_Notes.csv", encoding = "ISO-8859-1")
datasetInPandasDatFrame['Entities'] = np.nan
# dfWithNotes = datasetInPandasDatFrame["Notes/Questions"]
# dfWithNotes = datasetInPandasDatFrame["Notes/Questions"].dropna().drop_duplicates()
# dfWithNotes = datasetInPandasDatFrame.dropna().drop_duplicates()

# For Logistic notes
dfWithNotes = datasetInPandasDatFrame["Notes/Questions"].fillna('u')

# For Epic Notes
# dfWithNotes = datasetInPandasDatFrame["NOTE"].fillna('u')


# dfWithNotes.reset_index(drop=True)


listOfNotes = dfWithNotes.tolist()
print(listOfNotes)
# dfWithNotes = dfWithNotes.to_frame()
count = 0
for note in listOfNotes:
  # print(note)
  text = note
  doc = cat(text)
  datasetInPandasDatFrame.iloc[count,15]= str(doc.ents)


  # dfWithNotes['Entities']= (doc.ents)
  # dfWithNotes.loc[count,"Entities"]= (doc.ents)
  count += 1
# print(dfWithNotes.iloc[0,2])
print(datasetInPandasDatFrame)
datasetInPandasDatFrame.to_csv('Logistic_Notes_Entities.csv')
# datasetInPandasDatFrame.to_csv('Epic_Notes_Entities.csv')
# print(dfWithNotes)

text = "EVERY DAY, 7 CIGARETTES/DAY smoking smoker"

doc = cat(text)
# print(doc.ents)


print('\n','#'*10, 'CUID')
# If we want to see the CUI (ID) for each entity
for ent in doc.ents:
    print(ent, " - ", ent._.cui)

print('\n','#'*10, 'CUID-names')
# If we want to see the CUI (ID) for each entity
for ent in doc.ents:
    print(ent, " - ", cdb.cui2pretty_name[ent._.cui])

print('\n','#'*10, 'TUID')
    # To show semantic types for each entity
for ent in doc.ents:
  print(ent, " - ", ent._.tui)

print('\n','#'*10, 'TUID-name')
 # To show names for semenatic types
for ent in doc.ents:
  print(ent, " - ", cdb.tui2name[ent._.tui])

# We can also show the entities in a nicer way using displacy form spaCy
#from spacy import displacy
#displacy.render(doc, style='ent', jupyter=True)


# print(cdb.cui2pretty_name('C0020538'))
