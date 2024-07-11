import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re
import yaml

from pathlib import Path
from scipy.signal import resample

from functools import partial
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self,ulf_filepath,patient_csv,config):
        self.ulf_filepath = ulf_filepath
        self.patient_csv = patient_csv
        
        with open(config) as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        if self.config['resample'] == 'numpy':
            self.resample = partial(self.np_pad_resample,max_length = self.config['cutoff_length'])
        elif self.config['resample'] == 'scipy':
            self.resample = partial(self.scipy_resample,max_length = self.config['cutoff_length'])
        
        self.scaler = self.minmax_scaler

    def parse_mvnx_file(self,file,use_xzy = True):
        tree = ET.parse(file)
        root = tree.getroot()
        
        namespace = re.match(r'\{.*\}',root.tag).group(0)
        subject = root.find(namespace + 'subject')
        joints = subject.find(namespace + 'joints')
        label = []
        for joint in joints:
            name = joint.attrib['label']
            label.extend([name + '_x',name + '_y', name + '_z'])
        
        frames = subject.find(namespace + 'frames')
        ZXY = []
        XZY = []
        for frame in frames:
            jointangle = frame.find(namespace + 'jointAngle')
            ZXY.append(float(num) for num in jointangle.text.split(' '))
            jointangleXZY = frame.find(namespace + 'jointAngleXZY')
            XZY.append(float(num) for num in jointangleXZY.text.split(' '))

        df_zxy = pd.DataFrame(np.array(ZXY),columns=label)
        df_xzy = pd.DataFrame(np.array(XZY),columns=label)

        if use_xzy:
            df_zxy[['jRightShoulder_x','jRightShoulder_y','jRightShoulder_z','jLeftShoulder_x','jLeftShoulder_y','jLeftShoulder_z']] = df_xzy[['jRightShoulder_x','jRightShoulder_y','jRightShoulder_z','jLeftShoulder_x','jLeftShoulder_y','jLeftShoulder_z']]
        
        return df_zxy

    def get_dataset(self,test_patient  = set(),onehot_encode = True):
        TRAIN_DATA = []
        TEST_DATA = []


        for type in self.config['types']:
            for patient in os.listdir(self.ulf_filepath,type):
                for file in os.listdir(self.ulf_filepath,type,patient):
                    subject , task , hand = Path(file).stem.split('_')
                    df = self.parse_mvnx_file(os.path.join(self.ulf_filepath,type,patient,file))

                    if len(df) > self.config['cutoff_length']: continue

                    # Resample to max_length
                    temp_data = df[self.config['features']].to_numpy().T
                    for i in range(len(self.config['features'])):
                        temp_data[i] = self.scaler(self.resample(temp_data[i]))
                        
                    
                    assert temp_data.shape == (len(self.config['features']),self.config['cutoff_length']), f"Error while parsing MVNX data, expected shape {(len(self.config['features']),self.config['cutoff_length'])}"
                    temp_label = np.argwhere(self.config['tasks'] == task) if onehot_encode else task

                    if subject in test_patient: TEST_DATA.append([temp_data,temp_label])
                    else: TRAIN_DATA.append([temp_data,temp_label])
        
        
        return DataLoader(TRAIN_DATA,self.config['batch_size'],shuffle=True), DataLoader(TEST_DATA,self.config['batch_size'],shuffle=True)
    

    

    def np_pad_resample(self,sequence,max_length):
        assert max_length >= len(sequence), "Sequence Length Should not be longer than the Maximum Length."
        pad = max_length - len(sequence)
        pad_width = (pad // 2, pad // 2) if pad % 2 == 0 else ((pad-1) // 2, (pad-1) // 2 + 1)
        resample_sequence = np.pad(sequence,pad_width,mode='edge')
        return resample_sequence
    
    def scipy_resample(self,sequence,max_length):
        return resample(sequence,max_length)
 

    def minmax_scaler(self,sequence):
        return (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))









