import json
import sys
import os

numb = int(sys.argv[1])
print(numb)

json_name = 'pypoptim/configs/real_data_inactivation/M2_real.json'
dirname = "data/real/inactivation/"
output_folder_name = "../../../results/M2_real_inact/"

with open(json_name) as data_file:
    data = json.load(data_file)


phenotype =  os.path.join("../../../",dirname, "start.csv")

names = os.listdir(dirname)
name_base = ".".join(names[numb].split(".")[:-1])

data['experimental_conditions']['trace']['filename_phenotype'] = phenotype.replace('start', name_base)
data['output_folder_name'] = os.path.join(output_folder_name, name_base)

with open(json_name, 'w') as outfile:
    json.dump(data, outfile, indent=4)
