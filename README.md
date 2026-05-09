# Genetic Algorithm for Feature Selection

A Python implementation of a Genetic Algorithm (GA) for automated feature selection in binary classification tasks.

## Authors
First-year Computer Science students, Applied Sciences Faculty, Ukrainian Catholic University (UCU)

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/MasterDay3.png" width="120" style="border-radius:50%"><br>
      <a href="https://github.com/MasterDay3"><b>@MasterDay3</b></a><br>
      <b>Zabulskyi Mykola</b><br>
       data preparation and dataset management, testing and validation of project components, result visualization, and implementation of argument parsing for convenient project configuration and execution.
    </td>
    <td align="center">
      <img src="https://github.com/pmykhailyk-dot.png" width="120" style="border-radius:50%"><br>
      <a href="https://github.com/pmykhailyk-dot"><b>@pmykhailyk-dot</b></a><br>
      <b>Pelyno Mykhailo</b><br>
      implementation of the machine learning part and the fitness function, including model integration, result evaluation, and connection of the ML pipeline with the optimization process.
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="https://github.com/alinaytsk.png" width="120" style="border-radius:50%"><br>
      <a href="https://github.com/alinaytsk"><b>@alinaytsk</b></a><br>
      <b>Yatsko Alina</b><br>
      implementation of the genetic algorithm, including population evolution logic, selection, mutation, crossover mechanisms, and overall algorithm optimization.
    </td>
    <td align="center">
      <img src="https://github.com/havronskasoph.png" width="120" style="border-radius:50%"><br>
      <a href="https://github.com/havronskasoph"><b>@havronskasoph</b></a><br>
      <b>Havronska Sofia</b><br>
      implementation of the main program function, coordination of all project modules, and ensuring proper interaction between system components. Helped structure the overall workflow and maintain project consistency.
    </td>
  </tr>
</table>

---

## Mentor

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/draklowell.png" width="120" style="border-radius:50%"><br>
      <a href="https://github.com/draklowell"><b>@draklowell</b></a><br>
      <b>Andriy Kryvyi</b><br>
      Legend
    </td>
  </tr>
</table>

---

## Overview
This project applies a Genetic Algorithm to select the most relevant features from a dataset, reducing dimensionality while maintaining or improving model performance. The algorithm evolves a population of binary chromosomes, where each bit represents whether a feature is included or excluded.

## How It Works
1. **Initialization** — random population of binary chromosomes is generated
2. **Fitness Evaluation** — each chromosome is scored using cross-validated ROC-AUC with a penalty for using too many features
3. **Selection** — tournament selection picks the best candidates
4. **Crossover** — single-point crossover produces new children
5. **Mutation** — random bit flips introduce diversity
6. **Elitism** — best chromosome is always preserved to the next generation
7. **Early Stopping** — halts if no improvement is found for N generations

## Project Structure
```
├── main.py                  # Genetic algorithm loop and entry point
├── models.py                # Model definition and fitness function
├── genetic_algorithm.py     # GA operators: init, selection, crossover, mutation
├── data.py                  # Dataset loading, preprocessing, train/test split
├── visualization.py         # Plotting GA results
├── test.py                  # Runtime estimation utility
└── datasets/                # CSV datasets
```

## Requirements
```
numpy
pandas
scikit-learn
joblib
matplotlib
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Set your dataset path in `data.py`:
```python
FILENAME = 'datasets/your_dataset.csv'
```

2. Tune the algorithm parameters in `main.py`:
```python
N_GENERATION = 100
PENALTY = 0.01
```

3. Run:
```bash
python main.py
```

Or use CLI arguments:
```bash
python main.py --generations 50 --penalty 0.05 --population-size 30 --model logistic --scoring roc_auc
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `N_GENERATION` | 100 | Number of generations |
| `PENALTY` | 0.01 | Penalty for using too many features |
| `population_size` | 30 | Number of chromosomes per generation |
| `crossover_rate` | 0.8 | Probability of crossover |
| `mutation_rate` | 0.05 | Probability of bit flip |
| `tournament_k` | 3 | Tournament selection size |
| `cv` | 3 | Cross-validation folds |
| `scoring` | roc_auc | Fitness scoring metric |

## Requirements for Dataset
- CSV format
- Target variable must be the **last column**
- Target must be binary: `0` and `1`
- Any categorical columns are automatically encoded

## Example Results
<img width="1280" height="753" alt="telegram-cloud-photo-size-2-5474239975911006146-y" src="https://github.com/user-attachments/assets/a6259ed3-1536-4737-a58b-019423a9f3e4" />
