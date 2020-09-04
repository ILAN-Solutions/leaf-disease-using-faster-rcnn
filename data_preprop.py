import pandas as pd
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--path", dest="data_path", help="Path to data.")
parser.add_option("-o", "--output", dest="output_file", help="Output file (.txt format)", default = "output.txt")
(options, args) = parser.parse_args()

if not options.data_path:
	parser.error('Error: path to data must be specified. Pass --path to command line')



train =  pd.DataFrame()
data = []
path = options.data_path
total_files = os.listdir(path)
print("Total number of images:")
print(len(total_files))

for filename in tqdm(total_files):
    ext = os.path.splitext(os.path.join(path,filename))[1]

    if ext=='.jpg' or ext=='.png' or ext=='.PNG' or ext =='.JPG':

        xml_file=(os.path.splitext(os.path.join(path,filename))[0]+'.xml')
        etree = ET.parse(xml_file).getroot()
        for elm in etree.iter('bndbox'):
            xmin = int(elm.find('xmin').text)
            xmax = int(elm.find('xmax').text)
            ymin = int(elm.find('ymin').text)
            ymax = int(elm.find('ymax').text)
            for ele in etree.iter('name'):
                class_name = ele.text
                class_name = class_name.replace(" ", "_")



            tmp = path +'/' + filename + ',' + str(xmin)+',' + str(ymin)+',' + str(xmax)+',' + str(ymax)+',' + class_name
            data.append(tmp)
train['format'] = data
#print(data)
train.to_csv(options.output_file,header=None,index=None, sep=' ')

