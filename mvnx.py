import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re
import yaml
from tqdm import tqdm

from pathlib import Path
from scipy.signal import resample

from functools import partial


class PreprocessMVNX:
    def __init__(self,config):
        '''
        Output Training dataset shape should be (B,T,C) for tensorflow, (B,C,T) for pytorch
        '''
        self.config = config

        if self.config['resample'] == 'numpy':
            self.resample = partial(self.np_pad_resample,max_length = self.config['cutoff_length'])
        elif self.config['resample'] == 'scipy':
            self.resample = partial(self.scipy_resample,max_length = self.config['cutoff_length'])
        

    def parse_mvnx_file(self,file,use_xzy = True):
        '''
        Function for preprocessing jointAngle in mvnx file
        '''
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
        for frame in frames[3:]:
            jointangle = frame.find(namespace + 'jointAngle')
            ZXY.append([float(num) for num in jointangle.text.split(' ')])
            jointangleXZY = frame.find(namespace + 'jointAngleXZY')
            XZY.append([float(num) for num in jointangleXZY.text.split(' ')])

        df_zxy = pd.DataFrame(np.array(ZXY),columns=label)
        df_xzy = pd.DataFrame(np.array(XZY),columns=label)

        if use_xzy:
            df_zxy[['jRightShoulder_x','jRightShoulder_y','jRightShoulder_z','jLeftShoulder_x','jLeftShoulder_y','jLeftShoulder_z']] = df_xzy[['jRightShoulder_x','jRightShoulder_y','jRightShoulder_z','jLeftShoulder_x','jLeftShoulder_y','jLeftShoulder_z']]
        
        return df_zxy

    def get_dataset(self,path,test_patient  = set()):
        self.ulf_filepath = path

        TRAIN_DATA = []
        TEST_DATA = []
        TRAIN_LABEL = []
        TEST_LABEL = []

        for type in self.config['types']:
            print(f"Processing {type} Folder.")
            type_dir = os.path.join(self.ulf_filepath,type)
            for patient in tqdm(os.listdir(type_dir),desc="Patient",leave=False):
                patient_dir = os.path.join(type_dir,patient)
                for file in tqdm(os.listdir(patient_dir),desc="File",leave=True):
                    subject , task , hand = Path(file).stem.split('_')
                    df = self.parse_mvnx_file(os.path.join(patient_dir,file))

                    if len(df) > self.config['cutoff_length']: continue

                    # Resample to max_length
                    temp_data = df[self.config['features']].to_numpy() # T,C
                    resample_data = np.zeros((self.config['cutoff_length'],temp_data.shape[1]))

                    for i in range(temp_data.shape[1]):
                        resample_data[:,i] = self.resample(temp_data[:,i])

                    assert resample_data.shape == (self.config['cutoff_length'],len(self.config['features'])), f"Error while parsing MVNX data, expected shape {(len(self.config['features']),self.config['cutoff_length'])}, get shape {resample_data.shape}"
                    
                    temp_label = np.argwhere(task == self.config['tasks'])

                    if subject in test_patient: 
                        TEST_DATA.append(resample_data)
                        TEST_LABEL.append(temp_label)
                    else: 
                        TRAIN_DATA.append(resample_data)
                        TRAIN_LABEL.append(temp_label)


        return (np.array(TRAIN_DATA), np.array(TRAIN_LABEL).reshape(-1,1)), (np.array(TEST_DATA), np.array(TEST_LABEL).reshape(-1,1))
    

    

    def np_pad_resample(self,sequence,max_length):
        assert max_length >= len(sequence), "Sequence Length Should not be longer than the Maximum Length."
        pad = max_length - len(sequence)
        pad_width = (pad // 2, pad // 2) if pad % 2 == 0 else ((pad-1) // 2, (pad-1) // 2 + 1)
        resample_sequence = np.pad(sequence,pad_width,mode='edge')

        assert len(resample_sequence) == max_length
        return resample_sequence
    
    def scipy_resample(self,sequence,max_length):
        return resample(sequence,max_length)
 

    def minmax_scaler(self,sequence):
        return (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))






