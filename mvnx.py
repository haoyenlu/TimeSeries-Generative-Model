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
    def __init__(self,features, types, tasks, cutoff_length, resample):
        '''
        Output Training dataset shape should be (B,T,C) for tensorflow, (B,C,T) for pytorch
        '''
        self.features = features
        self.types = types
        self.tasks = tasks
        self.cutoff_length = cutoff_length
        self.resample =  resample

        if self.resample == 'numpy':
            self.resample_fn = partial(self.np_pad_resample,max_length = cutoff_length)
        
        elif self.resample== 'scipy':
            self.resample_fn = partial(self.scipy_resample,max_length = cutoff_length)
        
        else:
            raise Exception("Resample method not defined!")
        
    def _parse_mvnx_file(self,file,use_xzy = True):
        '''
        Processing jointAngle data in mvnx file
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
        XZY = [] # use XZY for shoulder joint angle to prevent gimbal lock
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
        '''
        Upper Limb Functionality Dataset
        
        Structure:
        Types -> Patient -> trial file
        '''
        TASKS = {task.upper():[] for task in self.tasks}
        TEST_PATIENT_TASKS = {task.upper():[] for task in self.tasks}

        print("Processing ULF Folder!")

        for type in self.config['types']:

            type_dir = os.path.join(path,type)
            for patient in tqdm(os.listdir(type_dir),desc="Patient",leave=False):

                patient_dir = os.path.join(type_dir,patient)
                for file in tqdm(os.listdir(patient_dir),desc="File",leave=True):

                    subject , task , data = self._get_data(file)

                    T,C = data.shape
                    if T > self.cutoff_length: continue

                    resample_data = np.zeros((self.cutoff_length,C))

                    for i in range(C):
                        resample_data[:,i] = self.resample_fn(data[:,i])

                    assert resample_data.shape == (self.cutoff_length,C), f"Error while parsing MVNX data, expected shape {(self.cutoff_length,len(self.features))}, get shape {resample_data.shape}"

                    task = task.upper()

                    if subject not in test_patient:
                        TASKS[task].append(resample_data)
                    else:
                        TEST_PATIENT_TASKS[task].append(resample_data)

        return TASKS, TEST_PATIENT_TASKS
    

    def _get_data(self,file) :# shape=(T,C)
        subject , task , hand = Path(file).stem.split('_')
        df = self._parse_mvnx_file(file)
        data = df[self.config['features']].to_numpy() 
        return subject, task, data

    

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






