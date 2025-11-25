# 🏎️ Formula 1 Race Result Prediction

This repository contains the **deep learning** part of my Computer Engineering graduation project.

The goal of the project is to **predict final race positions** using only the **early laps** of a Formula 1 race.
In the full project, we experimented with both **machine learning** and **deep learning** models.
This repo focuses on a **TimeDistributed LSTM** architecture, which achieved our best performance.

# 🎓 Project Background

* **Type:** Bachelor graduation project (Computer Engineering)
* **Task:** Predict each driver’s final position in a race
* **Approach:** Sequence modelling with a TimeDistributed LSTM
* **Key idea:**
  > Use as few laps as possible while keeping prediction error low.

# 🔍 Problem Definition

Given the first part of a race (lap-by-lap features for each driver), the model learns to predict the final race order:
* **Input:** For each driver, a sequence of laps with multiple numerical features
* **Output:** Final race position for each driver in that race <br>
This is treated as a **regression problem on positions**, evaluated mainly with **Mean Absolute Error (MAE).**

# 🧾 Data

We use two main sources for data collection and analysis:

* **🏁 FastF1** – Python library for F1 timing & telemetry data
* **📊 Ergast Developer API** – Historical Formula 1 results & metadata <br>
From these sources, we build lap-by-lap data per driver.

### 🔢 Features

For each lap and each driver, we use the following **9 features**:

* ```LapTime``` – lap time
* ```TyreLife``` – number of laps on the current tyre
* ```FreshTyre``` – indicator for a new/fresh tyre (e.g. 0/1)
* ```SpeedST``` – speed-related metric (e.g. speed trap)
* ```LapNumber``` – current lap number in the race
* ```Position``` – race position at that lap
* ```SpeedI1``` – speed-related metric around the first timing point / sector
* ```SpeedI2``` – speed-related metric around the second timing point / sector
* ```SpeedFL``` – speed-related metric near the finish line

These 9 features form the **feature dimension** in our tensors.

### 🧹 Data Preparation Challenges

The hardest part of the project was cleaning and preparing the data:

* Handling **missing laps** and **DNFs**
* Aligning all drivers within the same race
* Ensuring sequences have consistent lengths (10 / 20 / 30 / 40 laps)
* Splitting train / test at **race level** (not randomly per lap)

### 📈 Race position diagram (concept)

You can think of a race as a set of lines showing how each driver’s position changes over time:

<img width="1145" height="547" alt="download" src="https://github.com/user-attachments/assets/eb5c3538-4d9e-47a4-99d1-4bf6b794f644" />

## 📐 Data Shape & Lap Windows

We experimented with different lap windows to see how much of the race we need to observe:
* First **10** laps
* First **20** laps
* First **30** laps
* First **40** laps

For each window, the general dataset shape is:
```bash
X: (num_races, num_drivers, num_laps_window, num_features)
y: (num_races, num_drivers)
```

In the final configuration (using **40 laps**, which gave the best performance), the shapes are:
```bash
X_train.shape  # (102, 20, 40, 9)
y_train.shape  # (102, 20)
X_test.shape   # (26, 20, 40, 9)
y_test.shape   # (26, 20)
```

For ```X_train```:
* **102** → number of races in the training set
* **20** → number of drivers per race
* **40** → number of laps used from each race (lap window = 40)
* **9** → number of features per lap (the list above)

So:
> 🔹 **One race** → a 3D tensor of shape (**20 drivers, 40 laps, 9 features**) <br>
> 🔹 **Full dataset** → a batch of 102 such race tensors in ```X_train```

```y_train``` has shape **(102, 20)**:
* For each race, we have **20 targets** → the final positions of all 20 drivers.

