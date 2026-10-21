# GMM Image Segmentation

An educational notebook that groups brain CT pixels by intensity using a Gaussian Mixture Model. Intensity clusters are **not validated hemorrhage labels**: the brightest cluster can include skull and other structures. Detection claims require evaluation against expert annotations.

![GMM example](GMM_Example.jpg)

## Setup

Use Python 3.12 (validated):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select this environment's Python kernel in your notebook editor. No web server is required. Dependencies are pinned directly; transitive dependencies are not locked.

## Download data with Kaggle

Create a Kaggle API token in your Kaggle account settings. Configure `KAGGLE_API_TOKEN` securely outside the repository (cloud users: environment settings). Never commit credentials or paste them into the notebook. The Kaggle CLI also supports external credential configuration. On cloud machines with a read-only home directory, the script uses `/workspace/.config/kaggle` for writable configuration. Set `KAGGLE_CONFIG_DIR` explicitly to reuse another configuration directory.

Download the [Head CT hemorrhage dataset](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) outside Git:

```bash
export GMM_IMAGE_DIR=/workspace/datasets/head-ct-hemorrhage
bash scripts/download_data.sh
```

Choose another absolute directory for local use. Run the notebook with the same `GMM_IMAGE_DIR`; its default is the cloud path above. PNG files are discovered recursively, including nested extraction directories. Grayscale, RGB, and RGBA inputs are supported. Missing directories or empty PNG collections produce an actionable error.

Cloud network settings must allow `kaggle.com`, `www.kaggle.com`, `api.kaggle.com`, and `storage.googleapis.com`. Follow Kaggle's dataset access and license terms. The Kaggle API download and loading of all 200 PNGs were validated in the cloud environment. Full notebook execution, including overlays, was validated on two real CT images; medical detection accuracy has not been evaluated.

## Checks

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

The tests cover image loading, invalid inputs, intensity ordering, data discovery failures, and complete notebook execution with synthetic images. They do not measure medical detection accuracy.
