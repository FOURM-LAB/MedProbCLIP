# MedProbCLIP: Probabilistic Adaptation of Vision–Language Foundation Model for Reliable Radiograph–Report Retrieval
<div align="center">

  [![WACV Workshop Paper](https://img.shields.io/badge/WACV'26-Workshop-blue)](https://aaai.org/conference/aaai/aaai-26)&nbsp;&nbsp;
  [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://www.gb-liang.com/projects/betarisk)&nbsp;&nbsp;
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
    todo...

4.  **Prepare Data Files:**
    todo...

5.  **Run the Training Script:**
    todo

### 📜 Citation
```
@inproceedings{elallaf2026betarisk,
  title={MedProbCLIP: Probabilistic Adaptation of Vision–Language Foundation Model for Reliable Radiograph–Report Retrieval},
  author={Elallaf, Ahmad and Zhang, Yu and Yang, Joeng and Lee, Young and Cao, Zechun and Liang, Gongbo},
  booktitle={WACV Workshop},
  year={2026}
}
```

#### 📄 License
Distributed under the MIT License. See ```LICENSE``` for more information.
