# LCNN countermeasure for ASVspoof 2019 LA

PyTorch project for training a Light CNN countermeasure on the Logical Access
partition of ASVspoof 2019. The project is based on
[`Blinorot/pytorch_project_template`](https://github.com/Blinorot/pytorch_project_template).

The initial baseline uses:

- 16 kHz mono audio;
- a 512-point log-power STFT (20 ms window, 10 ms shift);
- 750 frames with random training crops and deterministic evaluation crops;
- an LCNN with Max-Feature-Map activations;
- cross-entropy loss and Adam;
- validation EER for checkpoint selection;
- Weights & Biases experiment tracking.

The LCNN implementation was reconstructed from the architecture descriptions in
the papers listed under [References](#references); no third-party LCNN
implementation is used.

## Installation

Python 3.10 or 3.11 is recommended. PyTorch packages in `requirements.txt` are
the CUDA-independent versions; on Kaggle or another GPU machine, keep the
preinstalled compatible PyTorch build when appropriate.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
pre-commit install
```

Run the checks:

```bash
python3 -m pytest
python3 -m compileall -q train.py inference.py src
```

## Dataset layout

Download the ASVspoof 2019 Logical Access data separately. Do not commit it.
The commands below assume the following layout:

```text
/path/to/asvspoof/
├── ASVspoof2019_LA_train/flac/*.flac
├── ASVspoof2019_LA_dev/flac/*.flac
└── ASVspoof2019_LA_eval/flac/*.flac

/path/to/protocols/
├── ASVspoof2019.LA.cm.train.trn.txt
├── ASVspoof2019.LA.cm.dev.trl.txt
└── ASVspoof2019.LA.cm.eval.trl.txt
```

Labels use `bonafide=1` and `spoof=0`. Consequently, a larger submitted score
means that an utterance is more likely to be bona fide.

## One-batch test

Before full training, verify that the complete pipeline can overfit eight fixed
development examples:

```bash
python3 train.py -cn=lcnn_one_batch \
  data_dir=/path/to/asvspoof \
  protocol_dir=/path/to/protocols
```

The loss should approach zero and the validation accuracy should approach one.
The one-batch configuration logs offline and must not be used as the final
experiment.

## Training

Authenticate with W&B and start the baseline:

```bash
wandb login
python3 train.py -cn=lcnn \
  data_dir=/path/to/asvspoof \
  protocol_dir=/path/to/protocols \
  writer.run_name=lcnn-stft-ce-seed1
```

Common overrides:

```bash
# Use a different seed.
python3 train.py -cn=lcnn ... trainer.seed=10 writer.run_name=lcnn-seed10

# Reduce memory/worker use.
python3 train.py -cn=lcnn ... dataloader.batch_size=16 dataloader.num_workers=2
```

The baseline intentionally selects checkpoints using **development EER**, not
evaluation EER. Checkpoints are stored below `saved/<run_name>/`.

## Evaluation and submission

Run inference with the best checkpoint:

```bash
python3 inference.py -cn=lcnn_inference \
  data_dir=/path/to/asvspoof \
  protocol_dir=/path/to/protocols \
  inferencer.from_pretrained=/absolute/path/to/model_best.pth
```

This writes `data/saved/submission/eval.csv`. It has no header and contains:

```text
utterance_id,bonafide_logit
```

Rename it to the required university username and validate it with the official
`grading.py` before submission.

## Project map

```text
src/datasets/asvspoof.py  protocol parsing and log-STFT extraction
src/model/lcnn.py         LCNN and Max-Feature-Map
src/loss/                 training objectives
src/metrics/eer.py        official-style EER computation
src/configs/lcnn*.yaml    training, one-batch, and inference recipes
tests/                    shape, protocol, MFM, and EER checks
```

Only the reusable training, logging, checkpointing, and configuration
infrastructure is retained from the original template.

## Reproducibility checklist

- Keep train, dev, and eval protocols separate.
- Select hyperparameters and checkpoints only on dev.
- Record the random seed, package versions, GPU, and complete Hydra config.
- Keep W&B logs from the actual template training run.
- Export report plots from logged CSV data rather than screenshots.
- Never commit datasets, checkpoints, W&B credentials, or private tokens.

## References

- X. Wu et al., *A Light CNN for Deep Face Representation with Noisy Labels*,
  2018.
- G. Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019
  Challenge*, 2019.
- X. Wang and J. Yamagishi, *A Comparative Study on Recent Neural Spoofing
  Countermeasures for Synthetic Speech Detection*, 2021.
- ASVspoof consortium, *ASVspoof 2019 Evaluation Plan*.

## License

See [LICENSE](LICENSE). The underlying project template is MIT-licensed.
