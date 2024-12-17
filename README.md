# Time-series generative model for data augmentation using stroke patient upper limb data

This project involved using time-series generative model, specially diffusion model for data augmentation and increase the classification performance.

The dataset used in this project can be downloaded below:
https://zenodo.org/records/3713449

## Model

This project explored the used of time-series diffusion model with Unet backbone to synthesize patient jointangle data and perform classification on the task that the patient is performed.

## Result

After some explorations and experiments, we found that using diffusion model for data augmentation doesn't improve any classification performance.

### Original Prediction for patient 28
![P28-Original-Prediction](https://github.com/user-attachments/assets/7496e702-0919-42af-936c-c14f48b47f62)

### Augmented with Diffusion model 

![P28-Diffusion-Augmented-Prediciton](https://github.com/user-attachments/assets/338cd4d1-971f-4665-979f-7a41c85fd717)



## References:
Schwarz A, Bhagubai MMC, Nies SHG, Held JPO, Veltink PH, Buurke JH, Luft AR. Characterization of stroke-related upper limb motor impairments across various upper limb activities by use of kinematic core set measures. J Neuroeng Rehabil. 2022 Jan 12;19(1):2. doi: 10.1186/s12984-021-00979-0. Erratum in: J Neuroeng Rehabil. 2022 Jul 12;19(1):70. doi: 10.1186/s12984-022-01048-w. PMID: 35016694; PMCID: PMC8753836.
