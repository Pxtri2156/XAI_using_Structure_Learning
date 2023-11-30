import json 

labels_file = '/workspace/tripx/MCS/xai_causality/run/stad_cancer/labels.json'

# # labels = {0:'BRCA1',1:'CDK12',2:'KRAS',3:'NF1',4:'NRAS',5:'RB1',6:'TP53',7:'ZNF133',8:'OV'}
# # labels = {0:'ASXL1',1:'DNMT3A',2:'FLT3',3:'IDH1',4:'IDH2',5:'KIT',6:'KRAS',7:'NPM1',8:'PTPDC1',9:'PTPN11',10:'RUNX1',11:'SF3B1',12:'SMC1A',13:'TP53',14:'U2AF1', 15:'WT1', 16:'LAML'}
# labels = {0:'APC', 1:'ARID1A', 2:'ARID2', 3:'BCOR', 4:'CASP8', 5:'CDH1',
#           6:'CDKN2A', 7:'CTNNB1', 8:'DMD', 9:'ERBB2', 10:'FBXW7', 11:'KRAS',
#           12:'MUC6', 13:'PIK3CA', 14:'PTEN', 15:'RASA1', 16:'RHOA', 17:'RNF43', 18:'SMAD2',
#           19:'SMAD4', 20:'TP53', 21:'STAD'}

# with open(labels_file, 'w') as fi:
#     json.dump(labels, fi)
f = open(labels_file, 'r')
data = json.load(f)
print(data)