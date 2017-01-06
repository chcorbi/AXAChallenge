import os.path as osp
from datetime import timedelta


DATADIR = "data/"
DATAFILENAME = "train_2011_2012_2013.csv"
SUBFILENAME = "submission.txt"
DATAPATH = osp.join(DATADIR, DATAFILENAME)
SUBPATH = osp.join(DATADIR, SUBFILENAME)

SUBSET_ASSIGNEMENT = ['CMS', 'Crises', 'Domicile', 'Gestion', \
  'Gestion - Accueil Telephonique', 'Gestion Assurances', \
  'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', \
  'M\xc3\xa9dical', 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', \
  'Services', 'Tech. Axa', 'Tech. Inter', 'T\xc3\xa9l\xc3\xa9phonie', \
  'Tech. Total', 'M\xc3\xa9canicien', 'CAT', 'Manager', \
  'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']

ASSIGNEMENT = ['CAT', 'CMS', 'Crises', 'Domicile', 'Evenements', \
       'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex', \
       'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', \
       'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', \
       'Manager', 'M\xc3\xa9canicien', 'M\xc3\xa9dical', 'Nuit', \
       'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', \
       'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', \
       'T\xc3\xa9l\xc3\xa9phonie']

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
WEEK = timedelta(weeks=1)
MONTH = timedelta(days=30)
YEAR = timedelta(weeks=52)
