# Time-series generative model for data augmentation using stroke patient upper limb data

This project involved using time-series generative model, specially diffusion model for data augmentation and increase the classification performance.

The dataset used in this project can be downloaded below:
https://zenodo.org/records/3713449

## Model

This project explored the used of time-series diffusion model with Unet backbone to synthesize patient jointangle data and perform classification on the task that the patient is performed.

## Result

After some explorations and experiments, we found that using diffusion model for data augmentation doesn't improve any classification performance.

### Original and Synthetic Sample
![sample](https://github.com/user-attachments/assets/2eddc343-5cbf-4dd2-bc46-9773e4b23d28)

### Dimension Reduction
<p float="left" align="middle">
<img src="https://github.com/user-attachments/assets/c585c446-f967-4cb3-8888-ecae5eca8874" width="480" height="360">
<img src="https://github.com/user-attachments/assets/5ef01272-b3ae-451c-85ee-898bc281be46" width="480" height="360">
</p>






### Prediction for patient 28
<p float="left" align="middle">
<img src="https://github.com/user-attachments/assets/7496e702-0919-42af-936c-c14f48b47f62" width="480" height="360">
<img src="https://github.com/user-attachments/assets/338cd4d1-971f-4665-979f-7a41c85fd717" width="480" height="360">
</p>



## References:
Schwarz A, Bhagubai MMC, Nies SHG, Held JPO, Veltink PH, Buurke JH, Luft AR. Characterization of stroke-related upper limb motor impairments across various upper limb activities by use of kinematic core set measures. J Neuroeng Rehabil. 2022 Jan 12;19(1):2. doi: 10.1186/s12984-021-00979-0. Erratum in: J Neuroeng Rehabil. 2022 Jul 12;19(1):70. doi: 10.1186/s12984-022-01048-w. PMID: 35016694; PMCID: PMC8753836.
