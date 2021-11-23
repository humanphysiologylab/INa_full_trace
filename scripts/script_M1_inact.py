import json
import sys
import os

numb = int(sys.argv[1])

print(numb)

with open('pypoptim/configs/real_data_inactivation/M1_real.json') as data_file:
    data = json.load(data_file)

dirname = "data/real/inactivation/"
output_folder_name = "../../../results/M1_real_inact/"
phenotype =  os.path.join("../../../",dirname, "start.csv")

names = os.listdir(dirname)
data['experimental_conditions']['trace']['filename_phenotype'] = phenotype.replace('start', names[numb][:-4])
data['output_folder_name'] = os.path.join(output_folder_name, names[numb][:-4])

with open('pypoptim/configs/real_data_inactivation/M1_real.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
