# MedProbCLIP: Probabilistic Adaptation of Vision–Language Foundation Model for Reliable Radiograph–Report Retrieval
<div align="center">

  [![AAAI Paper](https://img.shields.io/badge/AAAI'26-xxxx-blue)](https://aaai.org/conference/aaai/aaai-26)&nbsp;&nbsp;
  [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://www.gb-liang.com/projects/betarisk)&nbsp;&nbsp;
  [![arXiv](https://img.shields.io/badge/arXiv-2511.04886-red?style=flat&label=arXiv)](https://arxiv.org/abs/2511.04886)&nbsp;&nbsp;

  [Ahmad Elallaf](https://www.linkedin.com/in/ahmadhamdy60344b234/),&nbsp;&nbsp;
  [Nathan Jacobs](https://jacobsn.github.io),&nbsp;&nbsp;
  [Xinyue Ye](https://geography.ua.edu/people/xinyue-ye),&nbsp;&nbsp;
  [Mei Chen](https://engr.uky.edu/people/mei-chen),&nbsp;&nbsp;
  [Gongbo Liang](https://www.gb-liang.com)

</div>

<!--Official implementation of ```BetaRisk```, a model that predicts uncertainty-aware roadway crash risk from satellite imagery by estimating a full Beta probability distribution.-->

### 📖 About The Project
Roadway safety models often produce a single, deterministic risk score, which fails to capture the model's own uncertainty--a critical flaw in safety--critical applications. ```BetaRisk``` addresses this by reframing risk assessment as a probabilistic learning problem. Instead of a single number, our model uses deep learning to analyze satellite imagery and predict a full **Beta probability distribution** of crash risk. This provides a richer, more trustworthy assessment by quantifying both the most likely risk and the model's confidence in that prediction.

<div align="center">
  <img src="imgs/case_study.png" width="768">
</div>

<div align="left">
  
**Figure 1.** A case study comparison for the San Antonio River Walk area. In the middle and right panels, each colored dot represents a sampled location where a risk score was predicted. **(Left) Ground Truth:** The locations of previous fatal crashes are shown as red diamonds. **(Middle) Baseline Model:** The baseline exhibits low recall, incorrectly assigning low-risk scores (blue) to many known crash sites at the sampled locations. **(Right) Our Model:** Our model demonstrates superior recall by correctly identifying more hazardous locations with elevated risk scores (yellow and orange), generating a more realistic and spatially coherent risk map.
</div>

### ✨ Key Features
- **Probabilistic Formulation:** Outputs the ```α``` and ```β``` parameters of a Beta distribution to capture both risk and uncertainty for every prediction.
- **Vision-Based:** Uses multi-scale satellite imagery as the sole input to learn the complex interplay of environmental risk factors.
- **State-of-the-Art Performance:** Achieves a 17-23% relative improvement in recall and superior model calibration (ECE) compared to strong baselines.

### 💡 Prediction Interpretability
A key advantage of ```BetaRisk``` is its ability to provide richer, more interpretable predictions than standard models. The table below demonstrates how a single, ambiguous score from a baseline model can correspond to **multiple, distinct scenarios** in ```BetaRisk```. For any given risk score, whether high (```0.85```), low (```0.15```), or medium (```0.50```), our model reveals the underlying confidence level, providing crucial context for decision-making that is unavailable in standard deterministic models.

<table align="center">
    <thead>
        <tr>
            <th>Risk Score</th>
            <th>Baseline Scenario</th>
            <th><code>BetaRisk</code> Scenarios (Ours)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" align="center"><code>0.85</code></td>
            <td rowspan="2">High Risk</td>
            <td><b>High confidence, high risk</b> (Risk: 0.85 ± 0.05 (α=30, β=5))</td>
        </tr>
        <tr>
            <td><b>Low confidence, high risk</b> (Risk: 0.85 ± 0.14 (α=5.1, β=0.9))</td>
        </tr>
        <tr>
            <td rowspan="2" align="center"><code>0.15</code></td>
            <td rowspan="2">Low Risk</td>
            <td><b>High confidence, low risk</b> (Risk: 0.15 ± 0.04 (α=5, β=30))</td>
        </tr>
        <tr>
            <td><b>Low confidence, low risk</b> (Risk: 0.15 ± 0.14 (α=0.9, β=5.1))</td>
        </tr>
        <tr>
            <td rowspan="3" align="center"><code>0.50</code></td>
            <td rowspan="3">Medium Risk or Very Uncertain Prediction</td>
            <td><b>High confidence, medium risk</b> (Risk: 0.50 ± 0.08 (α=20, β=20))</td>
        </tr>
        <tr>
            <td><b>Low confidence, medium risk</b> (Risk: 0.50 ± 0.15 (α=10, β=10))</td>
        </tr>
        <tr>
            <td><b>Very uncertain prediction</b> (Risk: 0.50 ± 0.25 (α=2, β=2))</td>
        </tr>
    </tbody>
</table>

### 🚀 Getting Started
To set up the project and run the training code, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FOURM-LAB/BetaRisk.git
    cd BetaRisk
    ```

2.  **Create a virtual environment and install dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights:**
    Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1fueEPJ-fJeEf3fSQ8kpVK1GCF--x8BqJ/view?usp=sharing).
    Save the downloaded file as `pre-trained_weights_mscm.pth` inside the `./ckpts/pre-trained_weights/` directory.
    You might need to create these directories if they don't exist:
    ```bash
    mkdir -p ckpts/pre-trained_weights/
    mv /path/to/downloaded_file.pth ckpts/pre-trained_weights/pre-trained_weights_mscm.pth
    ```

4.  **Prepare Data Files:**
    The dataset used in this project can be found from [MTSL-RoadRisk](https://www.gb-liang.com/projects/mtsl-roadrisk). 

5.  **Run the Training Script:**
    Once the dependencies are installed and pre-trained weights are in place, you can start the training process by running:
    ```bash
    python betarisk_train.py
    ```

### 📜 Citation
```
@inproceedings{elallaf2026betarisk,
  title={Beta Distribution Learning for Reliable Roadway Crash Risk Assessment},
  author={Elallaf, Ahmad and Jacobs, Nathan and Ye, Xinyue and Chen, Mei and Liang, Gongbo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

#### 📄 License
Distributed under the MIT License. See ```LICENSE``` for more information.
