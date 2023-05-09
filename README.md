# Semi-Supervised Federated Learing for Keyword Spotting
[ICME 2023] This is an implementation of [Semi-Supervised Federated Learing for Keyword Spotting]()
- An illustration of Semi-Supervised Federated Learning (SSFL) for Keyword Spotting (KWS).
<p align="center">
<img src="/asset/ssfl.png">
</p>
- An illustration of leveraging unlabeled audio streams in Semi-Supervised Learning (SSL).
<p align="center">
<img src="/asset/semi.png">
</p>

## Requirements
See `requirements.txt`

## Instructions
 - Global hyperparameters are configured in `config.yml`
 - Use `make.sh` to generate run script
 - Use `make.py` to generate exp script
 - Use `process.py` to process exp results
 - Experimental setup are listed in `make.py` 
 - Hyperparameters can be found at `process_control()` in utils.py 
 - `modules/modules.py` defines Server and Client
    - sBN statistics are updated in `distribute()` of Server
    - global momemtum is used in `update()` of Server
    - fix and mix dataset are constructed in `make_dataset()` of Client
 - The data are split at `split_dataset()` in `data.py`
 
## Examples
 - Train SSL for SpeechCommandsV1 dataset (TCResNet18, $N_\mathcal{S}=250, Weak Augment: BasicAugment, Strong Augment: SpecAugment and MixAugment)
    ```ruby
    python train_classifier_semi.py --control_name SpeechCommandsV1_tcresnet18_250_basic=basic-spec_fix
    ```
 - Train FL (Alternate) for SpeechCommandsV2 dataset (TCResNet18, $N_\mathcal{S}=250, Augment: BasicAugment, $M=100$, $C=0.1$, Non-IID ( $Dir(0.1)$ ))
    ```ruby
    python train_classifier_ssfl.py --control_name SpeechCommandsV2_tcresnet18_250_basic_sup_100_0.1_non-iid-d-0.1
    ```
 - Train SemiFL for SpeechCommandsV1 dataset (TCResNet18, $N_\mathcal{S}=2500, Weak Augment: BasicAugment, Strong Augment: SpecAugment and MixAugment, $M=100$, $C=0.1$, IID)
    ```ruby
    python train_classifier_ssfl.py --control_name SpeechCommandsV1_tcresnet18_2500_basic=basic-spec_fix-mix_100_0.1_iid
    ```
 - Test SemiFL for SpeechCommandsV2 dataset (TCResNet18, $N_\mathcal{S}=2500, Weak Augment: BasicAugment, Strong Augment: BasicAugment, $M=100$, $C=0.1$, Non-IID ( $K=2$ ))
    ```ruby
    python test_classifier_ssfl.py --control_name SpeechCommandsV2_tcresnet18_2500_basic=basic_fix_100_0.1_non-iid-l-2
    ```
    
## Results
- Learning curves of SSL and SSFL for TC-ResNet18 with Speech Commands datasets and $N_\mathcal{L}=\{250,2500\}$.
<p align="center">
<img src="/asset/lc.png">
</p>
- Comparison between ‘Parallel’ and ‘Alternate’ training for heterogeneous on-device data.
<p align="center">
<img src="/asset/alter.png">
</p>

## Acknowledgements
*Enmao Diao 
Eric W. Tramel  
Jie Ding  
Tao Zhang*