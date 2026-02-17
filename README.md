# MedProbCLIP: Probabilistic Adaptation of Vision–Language Foundation Model for Reliable Radiograph–Report Retrieval
<div align="center">

  [![WACV Workshop Paper](https://img.shields.io/badge/WACV'26-Workshop-blue)](https://sites.google.com/view/lfmbio)&nbsp;&nbsp;
  [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://www.gb-liang.com/projects/medprobclip)&nbsp;&nbsp;
  [![arXiv](https://img.shields.io/badge/arXiv-26xx.xxxx-red?style=flat&label=arXiv)](https://arxiv.org/abs/26xx.xxxx)&nbsp;&nbsp;

  [Ahmad Elallaf](https://www.linkedin.com/in/ahmadhamdy60344b234/),&nbsp;&nbsp;
  [Yu Zhang](https://yuzhang03.github.io),&nbsp;&nbsp;
  [Yuktha Priya Masupalli](https://www.linkedin.com/in/yukthapriya/),&nbsp;&nbsp;
  [Jeong Yang](https://scholar.google.com/citations?user=W5ssJOYAAAAJ&hl=en),&nbsp;&nbsp;  
  [Young Lee](https://scholar.google.com/citations?user=0l7CEwYAAAAJ&hl=en),&nbsp;&nbsp;
  [Zechun Cao](https://zechuncao.com),&nbsp;&nbsp;
  [Gongbo Liang](https://www.gb-liang.com)

</div>

<!--Official implementation of ```MedProbCLIP```, a probabilistic vision–language learning framework for chest X-ray and radiology report representation learning and bidirectional retrieval.-->

### 📖 About This Project
Vision–language foundation models have emerged as powerful general-purpose representation learners with strong potentials for multimodal understanding, but their deterministic embeddings often fail to provide the reliability required for high-stakes biomedical applications. This work introduces ```MedProbCLIP```, a probabilistic vision–language learning framework for chest X-ray and radiology report representation learning and bidirectional retrieval. The proposed method models image and text representations as **Gaussian embeddings** through a **probabilistic contrastive objective** that explicitly captures **uncertainty and many-to-many correspondences** between radiographs and clinical narratives. A variational information bottleneck mitigates overconfident predictions, while MedProbCLIP employs multi-view radiograph encoding and multi-section report encoding during training to provide fine-grained supervision for clinically aligned correspondence, yet requires only a single radiograph and a single report at inference. 

### 💡 Why Probabilistic Contrastive Learning: The Hidden Many-to-Many Problem
Although datasets typically assign one caption to one image, real image–text relationships are rarely one-to-one. As shown in **Figure 1**, human raters often identify a single caption may accurately describe many visually similar images (or vice versa). These *unannotated positives* become **false negatives**, misleading standard contrastive learning models that assume a strict one-to-one alignment.

<div align="center">
  <img src="imgs/coco_false-negative-match.png" width="600">
</div>

<div align="left">  
  
**Figure 1.** Illustration of inherent many-to-many relationships in cross-modal datasets. Although MS-COCO annotates only a single caption as the positive match to one image (blue arrows), human raters often identify multiple additional plausible matches (pink dashed arrows). Such unannotated positives create false negatives that violate the one-to-one assumption commonly enforced in contrastive learning, motivating models capable of handling ambiguity and uncertainty in image–text alignment.
</div>


**This structural ambiguity exists even more strongly in medical imaging—reports summarize multiple views, findings overlap across studies, and similar pathologies appear in many radiographs, resulting a fundamental mismatch:**
<div align="center">
  
|Reality|Standard Contrastive Learning|
|:--:|:--:|
|Many-to-many relationships|Forced one-to-one alignment|
|Natural ambiguity|Overconfident similarity scores|
|Multiple plausible matches|Punished as negatives|
|Heterogeneous clinical evidence|Single deterministic embedding|
<div align="left">  

**```MedProbCLIP```** is motivated by addressing this mismatch requires models that can **represent uncertainty**, **model multiple plausible matches**, and avoid **overconfident errors**. 


### ✨ Evaluation Results
Todo...

### 🚀 Getting Started
To set up the project and run the training code, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FOURM-LAB/MedProbCLIP.git
    cd MedProbCLIP
    ```

2.  **Create a virtual environment and install dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights:**
    todo...

4.  **Prepare Data Files:**
    This project requires CSV files to define dataset splits (e.g., `train.csv`, `val.csv`, `test.csv`).

    *   **Expected CSV Structure:** Each row in the CSV files should correspond to a study and typically include (but is not limited to) the following columns:
        *   `study_id`: Unique identifier for the study.
        *   `image_paths`: A path or list of paths to the image files associated with the study. If multiple images, they should be separated by a delimiter (e.g., `;`).
        *   `view_names`: Corresponding view names for each image (e.g., `PA`, `LATERAL`).
        *   `findings`: The 'Findings' section of the radiology report.
        *   `impression`: The 'Impression' section of the radiology report.
        *   **Example Row (Illustrative):**
            ```csv
            study_id,image_paths,view_names,findings,impression
            s12345,/path/to/image1.jpg;/path/to/image2.jpg,PA;LATERAL,Lungs are clear. No acute cardiopulmonary abnormality.,Normal chest x-ray.
            ```

    *   **For MIMIC-CXR Dataset:**
        The **MIMIC-CXR dataset** is a restricted-access dataset. Please refer to the official PhysioNet website for detailed instructions on how to obtain access, download the data (specifically the MIMIC-CXR-JPG version).
        *   *Note: Scripts for generating these exact CSVs from the raw MIMIC-CXR metadata are not provided in this repository, as specific preprocessing might vary. Users are expected to generate these based on their downloaded MIMIC-CXR data, ensuring the columns match the `CXRStudyDataset` expectations.*

5.  **Run the Training Script:**
    To train the MedProbCLIP model, use the `medprobclip/medprobclip_train.py` script. The script supports various hyperparameters for customization.

    **Basic Usage:**
    Ensure data CSVs (e.g., `train.csv`, `val.csv`, `test.csv`) are prepared as described in the previous section.

    ```bash
    python medprobclip/medprobclip_train.py \
      --train_csv /path/to/mimic_train.csv \
      --val_csv /path/to/mimic_val.csv \
      --test_csv /path/to/mimic_test.csv \
      --out ./checkpoints_medprobclip
    ```
    *   **Data Paths:** Replace `/path/to/mimic_train.csv`, `/path/to/mimic_val.csv`, and `/path/to/mimic_test.csv` with the actual paths to the prepared CSV files.
    *   **Output Directory:** The `--out` argument specifies where model checkpoints and TensorBoard logs will be saved.
    *   **GPU Usage:** The script automatically utilizes available GPUs. For multi-GPU training, it defaults to `nn.DataParallel`.

### 📜 Citation
```
@inproceedings{elallaf2026medprobclip,
  title={MedProbCLIP: Probabilistic Adaptation of Vision–Language Foundation Model for Reliable Radiograph-Report Retrieval},
  author={Elallaf, Ahmad and Zhang, Yu and Yang, Joeng and Lee, Young and Cao, Zechun and Liang, Gongbo},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},
  month={March},
  year={2026}
}
```

#### 📄 License
Distributed under the MIT License. See ```LICENSE``` for more information.
