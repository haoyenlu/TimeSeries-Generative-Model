import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re
from tqdm import tqdm

from pathlib import Path
from functools import partial

class MvnxParser:
    def __init__(self,features, types, tasks, max_length, resample):
        '''
        Output Training dataset shape should be (B,T,C)
        '''
        self.features = features
        self.types = types
        self.tasks = tasks
        self.max_length = max_length
        self.resample =  resample
        self.resample_fn = partial(self.np_interp_resample,max_length=max_length)

        
    def _parse_jointangle(self,file,use_xzy = True):
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
            for column in ['jRightShoulder_x','jRightShoulder_y','jRightShoulder_z','jLeftShoulder_x','jLeftShoulder_y','jLeftShoulder_z']:
                df_zxy[column] = df_xzy[column]
        
        return df_zxy

    def process_folder(self,path):
        '''
        Upper Limb Functionality Dataset
        
        Structure:
        Types -> Patient -> trial file
        '''
        print("Start Processing ULF dataset!")

        TASK = dict()
        for type in self.types:
            if type not in TASK:
                TASK[type] = dict()

            type_dir = os.path.join(path,type)
            for patient in tqdm(os.listdir(type_dir),desc="Patient",leave=False):
                p , _ = patient.split('_')
                if p not in TASK[type]:
                    TASK[type][p] = dict()

                patient_dir = os.path.join(type_dir,patient)
                for file in tqdm(os.listdir(patient_dir),desc="File",leave=True):

                    subject , task , data = self._process_data_from_file(os.path.join(patient_dir,file))
                    assert subject == p

                    # Discard data exceed max length
                    T,C = data.shape
                    if T > self.max_length: continue

                    # Resample Data
                    resample_shape = (self.max_length,C)
                    resample_data = np.zeros(resample_shape)
                    for i in range(C):
                        resample_data[:,i] = self.resample_fn(data[:,i])

                    assert resample_data.shape == resample_shape, f"Resample Error, expected shape {resample_shape}, get shape {resample_data.shape}."

                    # Append Data
                    task = task.upper()
                    if task not in TASK[type][p]:
                        TASK[type][p][task] = []

                    TASK[type][p][task].append(resample_data)


        return TASK
    

    def _process_data_from_file(self,file) :# shape=(T,C)
        subject , task , hand = Path(file).stem.split('_')
        df = self._parse_jointangle(file)
        data = df[self.features].to_numpy() 
        return subject, task, data

    

    def np_pad_resample(self,sequence,max_length):
        assert max_length >= len(sequence), "Sequence Length Should not be longer than the Maximum Length."
        pad = max_length - len(sequence)
        pad_width = (pad // 2, pad // 2) if pad % 2 == 0 else ((pad-1) // 2, (pad-1) // 2 + 1)
        resample_sequence = np.pad(sequence,pad_width,mode='edge')

        assert len(resample_sequence) == max_length
        return resample_sequence
    
    def np_interp_resample(self,sequence,max_length):
        n = len(sequence)
        resample_sequence = np.interp(np.arange(max_length), np.linspace(0,max_length,n),sequence)
        return resample_sequence
 







