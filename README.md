# Machine Learning and Deep Learning - 01TXFSM
##  Final Project: First Person Action Recognition

The git contains the source code associated with the final project of the course Machine Learning and Deep Learning - 01TXFSM.
This code contains different approaches to the first person activity recognition task. 

The work is based on the [EgoRNN](https://arxiv.org/pdf/1807.11794.pdf) and [LSTA](https://arxiv.org/pdf/2002.03982v1.pdf) 
architectures, that are further enhanced with the use of a Self-Supervised Motion Segmentation task following the method
proposed in the paper of [Planamente et. al](https://arxiv.org/pdf/2002.03982v1.pdf).
#### Prerequisites

* Python 3.5 o
* Pytorch 1.7.0
  #### 

### Running
#### EgoRNN
* ##### RGB

  * ###### Stage 1
  * ```
    python main-run-rgb.py --dataset gtea_61 
    --stage 1 
    --trainDatasetDir ./dataset/gtea_61/split2/train 
    --outDir experiments 
    --seqLen 25 
    --trainBatchSize 32 
    --numEpochs 300 
    --lr 1e-3 
    --stepSize 25 75 150 
    --decayRate 0.1 
    --memSize 512
    ```
  * ###### Stage 2
  * ```
    python main-run-rgb.py --dataset gtea61 
    --stage 2 
    --trainDatasetDir ./dataset/gtea_61/split2/train 
    --outDir experiments 
    --stage1Dict best_model_state_dict.pth 
    --seqLen 25 
    --trainBatchSize 32 
    --numEpochs 150 
    --lr 1e-4 
    --stepSize 25 75 
    --decayRate 0.1 
    --memSize 512
    ```
* ##### **Flow**
* ```
  python main-run-flow.py --dataset gtea61 
  --trainDatasetDir ./dataset/gtea_61/split2/train 
  --outDir experiments 
  --stackSize 5 
  --trainBatchSize 32 
  --numEpochs 750 
  --lr 1e-2 
  --stepSize 150 300 500 
  --decayRate 0.5
  ```
* ##### **Two Stream**
* ```
  python main-run-twoStream.py --dataset gtea61 
  --flowModel ./models/best_model_state_dict_flow_split2.pth 
  --rgbModel ./models/best_model_state_dict_rgb_split2.pth 
  --trainDatasetDir ./dataset/gtea_61/split2/train 
  --outDir experiments 
  --seqLen 25 
  --stackSize 5 
  --trainBatchSize 32 
  --numEpochs 250 
  --lr 1e-2 
  --stepSize 1 
  --decayRate 0.99 
  --memSize 512
  ```
* ##### **EgoRNN + MS Task**
* ```
  python ego-rnn/main-run-MS.py --stage 2
  --trainDatasetDir ./dataset/gtea_61/split2/train
  --outDir ./drive/MyDrive/MS_Task_1/frame_16_E
  --stage1Dict ./drive/MyDrive/rgb_16/stage1/model_rgb_state_dict.pth
  --seqLen 16 
  --trainBatchSize 32
  --numEpochs 150
  --lr 1e-4
  --stepSize 50 100
  --decayRate 0.1
  --memSize 512
  --regression
  ```

#### LSTA

#### LSTA MS Task

#### LSTA RepFlow Layer

