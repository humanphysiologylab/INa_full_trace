import json
import sys
import os

numb = int(sys.argv[1])

print(numb)
with open('pypoptim/configs/real_data/M2_real.json') as data_file:
    data = json.load(data_file)
dirname = "data/real/activation/"
output_folder_name = "../../../results/M2_real/"
phenotype =  "../../../data/real/activation/start.csv"
names = os.listdir(dirname)
data['experimental_conditions']['trace']['filename_phenotype'] = phenotype.replace('start', names[numb][:-4])
data['output_folder_name'] = os.path.join(output_folder_name, names[numb][:-4])
with open('pypoptim/configs/real_data/M2_real.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
