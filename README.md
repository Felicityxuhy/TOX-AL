# muTOX-AL
Implementation of "muTOX-AL" in PyTorch
![image](https://github.com/Felicityxuhy/TOX-AL/blob/main/TOX-AL%20Structure.png)
This is the official implementation of muTOX-AL. In this work, we propose **muTOX-AL**, **a task-independent deep active learning framework**, which could accurately predict **molecule mutagenicity** with fewer samples. **The framework comprises four parts**: the feature extraction module, the backbone module, the uncertainty estimation module and the loss calculation module. In muTOX-AL, five features of the molecule are extracted by the feature extraction module, the backbone module is used to predict the mutagenicity of the molecule, a simple and efficient uncertainty estimation module is used for estimating the most valuable samples.

# Getting Started
## Environment
This implementation is based on the python 3.9.6 environment.
Please use command prompt(or Anaconda prompt) to install the appropriate python packages and make sure they are in the appropriate version
- torch >= 1.11.0
- numpy >= 1.21.5
- pandas >= 1.2.4
- scikit-learn >=1.0.2
- tqdm >= 4.61.2

## Dataset
**TOXRIC dataset**: The raw data used in this study was the C. Xu's Ames data collection provided within 'In silico Prediction of Chemical Ames mutagenicity',which is one of the commonly used data sets for developing the prediction models. The entire database was prepared as follows. Firstly, any inorganic molecules, that is, those without carbon atoms within the structure are removed. Secondly, the molecules with unspecified stereochemistry were removed. Thirdly, the molecules were standardized using the InChI key. Finally, duplicates were identified and removed using the InChI key across the data collection.
In total, 7486 compounds were used for the model building. The data sets contained 4196 mutagens and 3289 non-mutagens. This data was downloaded from https://toxric.bioinforai.tech/home.

If you have already downloaded the data and convert it to fingerprints using python Rdkit package(MACCS, pubchem, Rdkit2d etc.), put it in the `data` folder and change the path in `prepare_data.py`

## Train and Evaluate
To train a model and evaluate, you can run it on the terminal as:
```python
python TOX-AL_main.py
 ```
