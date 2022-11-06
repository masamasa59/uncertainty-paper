# Predictive Uncertainty in Deep Learning
Predictive uncertainty is important for humans to trust machine learning systems.
This repo has collected papers to help you understand uncertainty in Deep Learning with a minimum of effort.

This repo contains research papers on predictive uncertainty, quantification methods, applications, and user studies.

I have collected papers from the following international conferences: ICML, NeurIPS, AAAI, ICLR, IUI, CHI

Keywords: 
`Data(Aleatoric) Uncertainty`, `Model(Epistemic) Uncertainty`, `Bayesain Neural Networks`, `Deep Ensembles`, `Uncertainty Visualization`, `Human-AI-Interaction`, `interpretability`, `XAI`


# Survey
* Gawlikowski, Jakob, et al. "A survey of uncertainty in deep neural networks." arXiv preprint arXiv:2107.03342 (2021).[[Link]](https://arxiv.org/abs/2107.03342)

* Zhou, Xinlei, et al. "A Survey on Epistemic (Model) Uncertainty in Supervised Learning: Recent Advances and Applications." Neurocomputing (2021).[[Link]](https://doi.org/10.1016/j.neucom.2021.10.119)

* E. Hu ̈llermeier and W. Waegeman, “Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods,” Machine Learning, vol. 110, no. 3, pp. 457–506, 2021. [[Link]](https://arxiv.org/abs/1910.09457)
 
* Abdar, Moloud, et al. "A review of uncertainty quantification in deep learning: Techniques, applications and challenges." Information Fusion (2021).[[Link]](https://doi.org/10.1016/j.inffus.2021.05.008)

* Meredith Skeels, Bongshin Lee, Greg Smith, and George Robertson. "Revealing uncertainty for information visualization." In Proceedings of the working conference on Advanced visual interfaces (AVI '08).[[Link]](https://dl.acm.org/doi/10.1145/1385569.1385637)

# Quantification Methods
### Single deterministic methods 
* Mukhoti, Jishnu, et al. "Deterministic Neural Networks with Inductive Biases Capture Epistemic and Aleatoric Uncertainty." arXiv preprint arXiv:2102.11582 (2021).[[Link]](https://arxiv.org/pdf/2102.11582)

* A. Malinin and M. Gales, “Predictive uncertainty estimation via prior networks,” in Advances in Neural Information Processing Systems, 2018, pp. 7047–7058.[[Link]](https://proceedings.neurips.cc/paper/2018/file/3ea2db50e62ceefceaf70a9d9a56a6f4-Paper.pdf)

* Liu, Jeremiah Zhe, et al. "A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness." arXiv preprint arXiv:2205.00403 (2022). [[Link]](https://arxiv.org/pdf/2205.00403.pdf)

* Sensoy, Murat, Lance Kaplan, and Melih Kandemir. "Evidential deep learning to quantify classification uncertainty." Advances in Neural Information Processing Systems 31 (2018).[[Link]](https://proceedings.neurips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf)

* Wen, Yeming, Dustin Tran, and Jimmy Ba. "Batchensemble: an alternative approach to efficient ensemble and lifelong learning." arXiv preprint arXiv:2002.06715 (2020).[[Link]](https://arxiv.org/pdf/2002.06715) 

### Ensembles
* Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).[[Link]](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)

* Valdenegro-Toro, Matias. "Deep sub-ensembles for fast uncertainty estimation in image classification." arXiv preprint arXiv:1910.08168 (2019). [[Link]](https://arxiv.org/pdf/1910.08168)

### Bayesian methods
* Jospin, Laurent Valentin, et al. "Hands-on Bayesian neural networks--a tutorial for deep learning users." arXiv preprint arXiv:2007.06823 (2020). [[Link]](https://arxiv.org/abs/2007.06823)
* Uncertainty in Deep Learning (PhD Thesis)[[Link1]](https://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)[[Link2]](https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html)

* Izmailov, Pavel, et al. "What are Bayesian neural network posteriors really like?." International Conference on Machine Learning. PMLR, 2021. [[Link]](https://arxiv.org/abs/2104.14421)

* Fortuin, Vincent. "Priors in bayesian deep learning: A review." arXiv preprint arXiv:2105.06868 (2021).[[Link]](https://arxiv.org/abs/2105.06868)

### Test-time augmentation
* Ashukha, Arsenii, et al. "Pitfalls of in-domain uncertainty estimation and ensembling in deep learning." arXiv preprint arXiv:2002.06470 (2020).[[Link]](https://arxiv.org/pdf/2002.06470) 

* Shanmugam, Divya, et al. "When and why test-time augmentation works." arXiv e-prints (2020): arXiv-2011.[[Link]](https://arxiv.org/pdf/2011.11156.pdf)

### Embedding
* Postels, Janis, et al. "The hidden uncertainty in a neural networks activations." arXiv preprint arXiv:2012.03082 (2020).[[Link]](https://arxiv.org/pdf/2012.03082)

# Applications
### Healtcare
* Nair, Tanya, et al. "Exploring uncertainty measures in deep networks for multiple sclerosis lesion detection and segmentation." Medical image analysis 59 (2020): 101557. [[Link]](https://www.sciencedirect.com/science/article/pii/S1361841519300994?casa_token=xkmmdBQmXdgAAAAA:rDYDtqJ3WI7EXwAFXZWsVezsmi7vll8nYTVnw3pGNs2aEoUFIKuBjCVi5D7evvSaNdMxaLMDuQ)

* Dusenberry, Michael W., et al. "Analyzing the role of model uncertainty for electronic health records." Proceedings of the ACM Conference on Health, Inference, and Learning. 2020.[[Link]](https://dl.acm.org/doi/pdf/10.1145/3368555.3384457)

### Self-driving cars semantic segmentation
* Kendall, Alex, Vijay Badrinarayanan, and Roberto Cipolla. "Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding." arXiv preprint arXiv:1511.02680 (2015).[[Link]](https://arxiv.org/pdf/1511.02680.pdf?source=post_page---------------------------)

### Active learning (Annotation)
* J. Zeng, A. Lesnikowski, and J. M. Alvarez, “The relevance of bayesian layer positioning to model uncertainty in deep bayesian active learning,” arXiv preprint arXiv:1811.12535, 2018. [[Link]](https://arxiv.org/pdf/1811.12535.pdf)

### Out of distribution detection 
* Postels, Janis, et al. "The hidden uncertainty in a neural networks activations." arXiv preprint arXiv:2012.03082 (2020).[[Link]](https://arxiv.org/pdf/2012.03082)

### Satellite image classification
* Gawlikowski, Jakob, et al. "Out-of-distribution detection in satellite image classification." arXiv preprint arXiv:2104.05442 (2021). [[Link]](https://arxiv.org/pdf/2104.05442)

### Foundation Model
* Tran, Dustin, et al. "Plex: Towards Reliability using Pretrained Large Model Extensions." arXiv preprint arXiv:2207.07411 (2022).[[Link]](https://arxiv.org/pdf/2207.07411.pdf)

### Dataset shift detection
* Ovadia, Yaniv, et al. "Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift." Advances in neural information processing systems 32 (2019). [[Link]](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)

* Postels, Janis, et al. "On the practicality of deterministic epistemic uncertainty." arXiv preprint arXiv:2107.00649 (2021).[[Link]](https://arxiv.org/pdf/2107.00649.pdf)

### Computer vision
* Valdenegro-Toro, M. "I find your lack of uncertainty in computer vision disturbing." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021)[[Link]](https://openaccess.thecvf.com/content/CVPR2021W/LXCV/papers/Valdenegro-Toro_I_Find_Your_Lack_of_Uncertainty_in_Computer_Vision_Disturbing_CVPRW_2021_paper.pdf)

### Natural language generation 
* Xiao, Yijun, and William Yang Wang. "On hallucination and predictive uncertainty in conditional language generation." arXiv preprint arXiv:2103.15025 (2021).[[Link]](https://arxiv.org/pdf/2103.15025)

### Text Classification
* J. Van Landeghem, M. Blaschko, B. Anckaert and M. -F. Moens, "Benchmarking Scalable Predictive Uncertainty in Text Classification," in IEEE Access, vol. 10, pp. 43703-43737, (2022)[[Link]](https://ieeexplore.ieee.org/document/9761166)

### Token-level and Sequence-level
* Jakob Smedegaard Andersen, Tom Schöner, and Walid Maalej. "Word-Level Uncertainty Estimation for Black-Box Text Classifiers using RNNs." In Proceedings of the 28th International Conference on Computational Linguistics, (2020)[[Link]](https://aclanthology.org/2020.coling-main.484/)

### Abstruction
* Alexios Gidiotis and Grigorios Tsoumakas. 2022. Should We Trust This Summary? Bayesian Abstractive Summarization to The Rescue. In Findings of the Association for Computational Linguistics: ACL (2022) [[Link]](https://aclanthology.org/2022.findings-acl.325/)

* Malinin, Andrey, and Mark Gales. "Uncertainty estimation in autoregressive structured prediction." arXiv preprint arXiv:2002.07650 (2020).[[Link]](https://arxiv.org/pdf/2002.07650.pdf)
### Node classification, Link prediction
* Pal, Soumyasundar, et al. "Non parametric graph learning for Bayesian graph neural networks." Conference on uncertainty in artificial intelligence. PMLR, 2020.[[Link]](https://arxiv.org/abs/2006.13335)

### Multiple instance learning
* Pal, Soumyasundar, et al. "Bag Graph: Multiple Instance Learning using Bayesian Graph Neural Networks." Proc. AAAI Conf. on Artificial Intelligence. 2022.[[Link]](https://www.aaai.org/AAAI22Papers/AAAI-8577.PalS.pdf)

### Time series imputation
* Fortuin, Vincent, et al. "Gp-vae: Deep probabilistic time series imputation." International conference on artificial intelligence and statistics. PMLR, 2020.[[Link]](https://proceedings.mlr.press/v108/fortuin20a.html)

### Camouflaged object detection
* Zhang, Jing, et al. "Dense uncertainty estimation." arXiv preprint arXiv:2110.06427 (2021).[[Link]](https://arxiv.org/pdf/2110.06427.pdf)

### Safty reinforcement learning
* Sedlmeier, Andreas, et al. "Uncertainty-based out-of-distribution classification in deep reinforcement learning." arXiv preprint arXiv:2001.00496 (2019).[[Link]](https://arxiv.org/pdf/2001.00496)

### Knowledge Distillation
* Ferianc, Martin, and Miguel Rodrigues. "Simple Regularisation for Uncertainty-Aware Knowledge Distillation." arXiv preprint arXiv:2205.09526 (2022).[[Link]](https://arxiv.org/pdf/2205.09526.pdf)

### XAI
* Antorán, Javier, et al. "Getting a clue: A method for explaining uncertainty estimates." arXiv preprint arXiv:2006.06848 (2020). [[Link]](https://arxiv.org/abs/2006.06848)

### SHAP
* Shaikhina, Torgyn, et al. "Effects of Uncertainty on the Quality of Feature Importance Explanations." AAAI Workshop on Explainable Agency in Artificial Intelligence. 2021.[[Link]](https://umangsbhatt.github.io/reports/AAAI_XAI_QB.pdf)

### Transparency
* Bhatt, Umang, et al. "Uncertainty as a form of transparency: Measuring, communicating, and using uncertainty." Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society. 2021.[[Link]](https://dl.acm.org/doi/pdf/10.1145/3461702.3462571)

### Counterfactual explanations
* Schut, Lisa, et al. "Generating interpretable counterfactual explanations by implicit minimisation of epistemic and aleatoric uncertainties." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.[[Link]](http://proceedings.mlr.press/v130/schut21a/schut21a.pdf)

### Robustness 
* Nado, Zachary, et al. "Uncertainty Baselines: Benchmarks for uncertainty & robustness in deep learning." arXiv preprint arXiv:2106.04015 (2021). [[Link]](https://arxiv.org/pdf/2106.04015)

### Calibration
* Guo, Chuan, et al. "On calibration of modern neural networks." International Conference on Machine Learning. PMLR, 2017.[[Link]](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf)

* Minderer, Matthias, et al. "Revisiting the calibration of modern neural networks." Advances in Neural Information Processing Systems 34 (2021).[[Link]](https://proceedings.neurips.cc/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf)

### Heteroscedastic noise
* Collier, Mark, et al. "Correlated input-dependent label noise in large-scale image classification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. (2021)[[Link]](https://openaccess.thecvf.com/content/CVPR2021/papers/Collier_Correlated_Input-Dependent_Label_Noise_in_Large-Scale_Image_Classification_CVPR_2021_paper.pdf)

### Example difficality
* Baldock, Robert, Hartmut Maennel, and Behnam Neyshabur. "Deep learning through the lens of example difficulty." Advances in Neural Information Processing Systems 34 (2021).[[Link]](https://arxiv.org/pdf/2106.09647.pdf)

* D'souza, Daniel, et al. "A Tale Of Two Long Tails." arXiv preprint arXiv:2107.13098 (2021).[[Link]](https://arxiv.org/pdf/2107.13098.pdf)

* Swayamdipta, Swabha, et al. "Dataset cartography: Mapping and diagnosing datasets with training dynamics." arXiv preprint arXiv:2009.10795 (2020).[[Link]](https://aclanthology.org/2020.emnlp-main.746.pdf)
# User Study
### Human collaboration 
* Kivlichan, Ian D., et al. "Measuring and improving model-moderator collaboration using uncertainty estimation." arXiv preprint arXiv:2107.04212 (2021).[[Link]](https://arxiv.org/pdf/2107.04212.pdf)

### Uncertainty visualization
* Téo Sanchez, Baptiste Caramiaux, Pierre Thiel, Wendy E. Mackay. "Deep Learning Uncertainty in Machine Teaching." 27th Annual Conference on Intelligent User Interfaces, Mar 2022 [[Link]](https://hal.archives-ouvertes.fr/hal-03579448/document)

* McGrath, Sean, et al. "When does uncertainty matter?: Understanding the impact of predictive uncertainty in ML assisted decision making." arXiv preprint arXiv:2011.06167 (2020).[[Link]](https://arxiv.org/pdf/2011.06167)

* Tak, Susanne, Alexander Toet, and Jan van Erp. "The perception of visual uncertaintyrepresentation by non-experts." IEEE transactions on visualization and computer graphics 20.6 (2013): 935-943.[[Link]](https://ieeexplore.ieee.org/iel7/2945/6805245/06654171.pdf?casa_token=WVSULJ9m1-AAAAAA:xNSXStIa4nVhlv19ixLkiFJ9Dk_wvJReKFv2nTLVM92tozGs_4QUvBLlUjIWta3aAsFY22i5pg)

### Application
* Miriam Greis, Emre Avci, Albrecht Schmidt, and Tonja Machulla. "Increasing Users' Confidence in Uncertain Data by Aggregating Data from Multiple Sources." In Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems (CHI '17). [[Link]](https://doi.org/10.1145/3025453.3025998)

* Yunfeng Zhang, Q. Vera Liao, and Rachel K. E. Bellamy."Effect of confidence and explanation on accuracy and trust calibration in AI-assisted decision making." In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* '20).[[Link]](https://arxiv.org/pdf/2001.02114.pdf)

* Matthew Kay, Tara Kola, Jessica R. Hullman, and Sean A. Munson. "When (ish) is My Bus? User-centered Visualizations of Uncertainty in Everyday, Mobile Predictive Systems." In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems (CHI '16).[[Link]](https://idl.cs.washington.edu/files/2016-WhenIsMyBus-CHI.pdf)


# Datasets
### RETINA Benchmark
* Band, Neil, et al. "Benchmarking bayesian deep learning on diabetic retinopathy detection tasks." NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications. 2021.[[Link]](https://openreview.net/pdf?id=uJ2_JTpVCvc)
