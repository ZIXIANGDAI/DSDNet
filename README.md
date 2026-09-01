# DSDNet: A Lightweight Divide-and-Conquer Network via Deep-Shallow Feature Specialization for Deep-Sea Polymetallic Nodule Video Segmentation

**Zixiang Dai<sup>1</sup>, Xu Yang<sup>1,*</sup>, Yugang Ren<sup>2</sup>, Limin Zhu<sup>3</sup>, Lei Jia<sup>4</sup>**

<sup>1</sup> Institute of Marine Science and Technology, Shandong University, Qingdao 266237, China <sup>2</sup> National Deep Sea Center, Qingdao 266237, China <sup>3</sup> State Key Laboratory of Mechanical System and Vibration, Shanghai Jiao Tong University, Shanghai 200240, China <sup>4</sup> School of Control Sciences and Engineering, Shandong University, Jinan 250061, China

## Abstract

Deep-sea polymetallic nodule video segmentation is essential for fine-grained seabed resource assessment and intelligent underwater perception. However, dense co-occurrence among nodules leads segmentation models to over-rely on collective co-occurrence cues, weakening the representation of intrinsic nodule characteristics. Meanwhile, existing polymetallic nodule segmentation methods typically process consecutive frames independently, resulting in redundant computation and insufficient temporal dependency modeling. To address these challenges, we propose a lightweight **Deep-Shallow Divide-and-Conquer Network (DSDNet)**, which assigns specialized roles to deep and shallow features. For deep features, the **Grouped Temporal Dependency Modeling Module (GTDMM)** distributes deep feature extraction across consecutive frames. At each time step, GTDMM extracts only one compact group-wise feature and integrates it with cached group-wise features from previous frames, thereby reconstructing a complete high-level representation enriched with temporal information while reducing redundant computation. For shallow features, the **Core-Edge Prior-Guided Modulation Module (CPMM)** progressively modulates decoder features with core and edge structural priors in a coarse-to-fine manner, enhancing the representation of intrinsic nodule characteristics under dense co-occurrence. Experiments on a real-world deep-sea polymetallic nodule video dataset collected by the Jiaolong submersible show that DSDNet reaches **0.9573 Dice**, **0.9186 IoU**, and **155.56 FPS**, demonstrating its effectiveness and efficiency for resource-constrained offshore scenarios.

**Keywords:** Deep-sea polymetallic nodules; Dense co-occurrence; Group-wise feature learning; Temporal dependency modeling; Video semantic segmentation.

## DSDNet

<p align="center">
  <img src="image/Figure_1.png" width="950">
</p>

## Main Components

### GSH

<p align="center">
  <img src="image/Figure_2.png" width="850">
</p>

### TGFIB

<p align="center">
  <img src="image/Figure_3.png" width="850">
</p>

### CPGB / EPGB

<p align="center">
  <img src="image/Figure_4.png" width="850">
</p>

### CSMB

<p align="center">
  <img src="image/Figure_5.png" width="850">
</p>