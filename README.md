# Evolutionary Learning for Antarctic Ice-Sheet Modelling
This project, to be submitted as my honors-year research project, endeavours to utilize evolutionary machine learning algorithms to more efficiently predict changes in ice sheets.

> [!CAUTION]
> This project **is not complete**, and should be viewed as a work-in-progress.

## 1. Brief

Climate change is causing sea levels to rise significantly, which considerably influences the human economy and species' living. One of the direct mechanisms that contribute to the observed sea level rise is the melting of land ice. It is vital to have some insight into how these ice sheets changes in the future. Numerical models have been used to predict how fast ice sheet is melting. These numerical models are based on complex physical numerical models with numerous complex differential equations, which is computationally expensive and time-consuming. This project will develop new evolutionary learning models that can predict ice sheet change effectively and efficiently.

---

## 2. Gallery

<img src="https://github.com/sam-mata/evolution-antarctica/assets/49130157/5dbfcb89-9da2-49e5-acef-8f4f5bc827ff" width="30%"></img> <img src="https://github.com/sam-mata/evolution-antarctica/assets/49130157/f2969be0-28ef-4662-8755-34d694ccd538" width="30%"></img> <img src="https://github.com/sam-mata/evolution-antarctica/assets/49130157/186b299f-9781-4c2d-92db-4b002a3dea67" width="30%"></img> 

<img src="https://github.com/sam-mata/evolution-antarctica/assets/49130157/67651b2c-75fb-4a28-a3da-22f5600c7f81" width="45%"></img> <img src="https://github.com/sam-mata/evolution-antarctica/assets/49130157/217fdfe8-1401-4ccc-b0dd-4e71fc9db659" width="45%"></img> 

---

## 3. Installation

After [**cloning the repository**](x-github-client://openRepo/https://github.com/sam-mata/sealevelai) or [**downloading a zip version**](https://github.com/sam-mata/SeaLevelAI/archive/refs/heads/main.zip), the project requires several libraries to be installed _(which are detailed in [`requirements.txt`](requirements.txt))_, this can be done with the following command:
After [**cloning the repository**](x-github-client://openRepo/https://github.com/sam-mata/sealevelai) or [**downloading a zip version**](https://github.com/sam-mata/SeaLevelAI/archive/refs/heads/main.zip), the project requires several libraries to be installed _(which are detailed in [`requirements.txt`](requirements.txt))_, this can be done with the following command:

```python
    pip install -r requirements.txt
```

---

## 4. Structure

### 4.1 Data

This project uses datasets provided by [**Professor Nicholas Golledge**](https://people.wgtn.ac.nz/nick.golledge).

The datasets are results from current physical simulations, split into `.txt` files labeled by year _(ranging 86 years from 2015 to 2100)_. Each file represents 1 year, holding 8 features split across 3 types:

1. **POSITIONAL CONSTANTS**

    Positional data is encoded with two input features: **`x coordinate`** and **`y coordinate`**. These are constant over time, and unique to each sample.

2. **BOUNDARY CONDITIONS / INPUT FORCINGS**

    There are three temporally-evolving boundary conditions that can be used for model predictions: **`precipitation`**, **`air temperature`**, and **`ocean temperature`**.

3. **OUTPUTS**

    There are three outputs to be predicted: **`ice thickness`**, **`ice velocity`**, and **`ice mask`**. Ideally these will be predicted from the input forcings and positional data.

#### Domain Knowledge

Several notes of domain knowledge were left with the data, giving possible expectations with how the data and models should behave according to current scientific understandings.

-   Outputs are predicted to correlate most with **`ocean temperature`** compared to other input features.
-   It is predicted that the output responses will be lagged with respect to the input forcings _(ie the **`ice thickness`** might start changing years or even decades after a change in boundary conditions.)_.

> [!WARNING]
> Many features hold no measured value in some samples, with these being filled with `NaN` or `9.96920996839e+36`.

### 4.2 Repository

-   [`/.github`](/.github) - Directory containing repository management tools, including issue templates and formatting checks.
-   [`/data`](/data/) - Input datasets.
-   [`EDA.ipynb`](EDA.ipynb) - Exploratory-data-analysis _(EDA)_
-   [`/model`](/model/) - Model training and testing.
-   [`/scripts`](/scripts/) - Helper scripts for various utilities.
-   [`LICENSE`](LICENSE) - Standard MIT usage license.
-   [`requirements.txt`](requirements.txt) - Required Python libraries and versions for correct usage.
-   [`/out`](/out/) - Project artifacts *(images, gifs, charts etc)*
-   [`/reports`](/reports/) - Project proposal, preliminary report, and final report.
-   [`model-testing.ipynb`](model-testing.ipynb) - Initial model testing.

---

## 5. Authorship

All project development completed by [**Sam Mata**](https://www.sammata.nz/), with supervision from [**Dr Bach Nguyen**](https://people.wgtn.ac.nz/bach.nguyen) and [**Dr Bing Xue**](https://people.wgtn.ac.nz/bing.xue).

Datasets provided by [**Professor Nicholas Golledge**](https://people.wgtn.ac.nz/nick.golledge) with initial exploratory data analysis being completed by **Serafina Slevin**.

This project was completed under the [**Center for Data Science and Artificial Intelligence**](https://www.wgtn.ac.nz/cdsai) in collaboration with the [**Antarctic Research Centre**](https://www.wgtn.ac.nz/antarctic) at [**Victoria University of Wellington**](https://www.wgtn.ac.nz/).
