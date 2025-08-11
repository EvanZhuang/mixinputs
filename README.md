<h1 align="center">🎨 Mixture of Inputs (MoI) 🎨</h1>
<p align="center"><b>Text Generation Beyond Discrete Token Sampling</b>  
(<a href="https://arxiv.org/abs/2505.14827">arXiv</a>)</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/pypi/v/mixinputs?color=green">  
</p>

<p align="center">
  <img src="assets/MoI.svg" alt="MoI Illustration" width="600">
</p>

---

## What is Mixture of Inputs?

**Mixture of Inputs (MoI)** is a simple, training-free technique to improve autoregressive text generation.

In standard LLM inference, the model samples a token and discards the full distribution. MoI keeps both—it mixes the sampled token with the original distribution to retain more information.

MoI uses Bayesian estimation to compute mixing weights:

- Prior: the output distribution

- Observation: the sampled token

- Posterior: a smooth input replacing the one-hot vector

This lets the model use a **richer representation state** $\boldsymbol{h}_t$ as inputs throughout generation, improving coherence, reasoning, and code synthesis.

---

## Installation

Requires `vllm == 0.8.5` or `vllm == 0.8.5.post1`.

```bash
pip install vllm==0.8.5
pip install mixinputs
mixinputs setup
```

## Quick Start
To activate MoI, set the MIXINPUTS_BETA environment variable:

```bash
export MIXINPUTS_BETA=1.0
```
Then run your usual vLLM-based generation script. That's it!

## CLI Utilities

The patch is installed via injection to `usercustomize.py` by adding `import mixinputs` at the head.

We provide command-line tools to patch or unpatch your environment:

```bash
# Enable MoI
mixinputs setup

# Disable MoI
mixinputs cleanup
```

You can also enable it via adding `import mixinputs` in your script before you import vllm. The mixinputs will not activate without the environment variable `MIXINPUTS_BETA`.

## Configuration Options

| Variable         | Description                                       | Default |
| ---------------- | ------------------------------------------------- | ------- |
| `MIXINPUTS_BETA` | Controls the strength of the mixture input signal | `1.0`   |

Recommended range: 0.5 to 2.0.

Tune based on task/model—lower values emphasize the distribution, higher values keep more of the sample.

Make sure `enforce_eager=True` in your LLM initialization.

Set `MIXINPUTS_BETA=-100` to activate direct mixture, i.e., mixing with solely output distributions.

## Examples 

We have included 2 example usages with [AIME](/example/aime.sh) and [Count Down 4](/example/countdown.sh), after executing the bash you should see 59-60 Acc for CountDown4 (Nemotron-Super-49B) and ~80 Acc for AIME (QwQ-32B).

For external evaluation packages, simply set `export MIXINPUTS_BETA=<BETA_VALUE>` and use vllm as the backbone, will activate MoI.

Additionally, MoI also work with server mode, we provide an example [script](/example/serve.sh).

## Evaluations

| Model                   | Method               | Input Info.     | AIME (%) | CountDown4 (%) | GPQA-D (%) | LiveCodeBench (pass@1) | Avg (%) |
| ----------------------- | -------------------- | --------------- | -------: | -------------: | ---------: | ----------------------: | -------: |
| **QwQ-32B**             | Standard             | Output Token    |    77.78 |          79.25 |      58.08 |                  76.32  |    72.86 |
|                         | Direct Mixture       | Output Dist.    |    72.00 |          66.88 |      51.52 |                  53.42  |    60.96 |
|                         | **MoI**              | Token + Dist.   |    80.00 |          80.01 |      60.10 |                  76.51  |    74.15 |
|                         | *Gain vs. Standard*  |                 | **+2.22**|       **+0.76**| **+2.02**  |            **+0.19**    | **+1.29**|
| **Nemotron-Super-49B**  | Standard             | Output Token    |    54.89 |          56.93 |      60.60 |                  39.92  |    53.09 |
|                         | Direct Mixture       | Output Dist.    |    60.00 |          51.72 |      60.10 |                  16.04  |    46.97 |
|                         | **MoI**              | Token + Dist.   |    57.11 |          59.53 |      64.65 |                  40.50  |    55.45 |
|                         | *Gain vs. Standard*  |                 | **+2.22**|       **+2.60**| **+4.05**  |            **+0.58**    | **+2.36**|
| **Gemma-3-27B**         | Standard             | Output Token    |    25.56 |          56.51 |      46.97 |                  31.31  |    40.09 |
|                         | Direct Mixture       | Output Dist.    |    26.44 |          55.47 |      51.52 |                  31.99  |    41.36 |
|                         | **MoI**              | Token + Dist.   |    26.89 |          59.38 |      47.47 |                  32.87  |    41.65 |
|                         | *Gain vs. Standard*  |                 | **+1.33**|       **+2.87**| **+0.50**  |            **+1.56**    | **+1.56**|
| **DAPO-Qwen-32B**       | Standard             | Output Token    |    64.67 |          72.03 |      42.42 |                  54.01  |    58.28 |
|                         | Direct Mixture       | Output Dist.    |    62.67 |          67.19 |      37.88 |                  23.87  |    47.90 |
|                         | **MoI**              | Token + Dist.   |    64.44 |          78.75 |      42.93 |                  55.18  |    60.33 |
|                         | *Gain vs. Standard*  |                 | **–0.23**|       **+6.72**| **+0.51**  |            **+1.17**    | **+2.05**|


## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at y5zhuang@ucsd.edu.

## Citation

If you find our paper and code useful, please cite us:
```r
@article{zhuang2025textgen,
  title={Text Generation Beyond Discrete Token Sampling},
  author={Zhuang, Yufan and Liu, Liyuan and Singh, Chandan and Shang, Jingbo and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2505.14827},
  year={2025}
}
```
