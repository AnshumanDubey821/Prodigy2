# Customer Segmentation Using K-Means Clustering

## Task 02

Create a K-Means clustering algorithm to group customers of a retail store based on their purchase history.

## Project Overview

This project segments retail customers into different groups using the K-Means clustering algorithm. The model uses customer income and spending behavior to identify useful customer segments for marketing and business decisions.

The dataset used is `Mall_Customers.csv`, which contains customer demographic and spending information.

## Dataset

The dataset includes the following columns:

- `CustomerID`: Unique customer identifier
- `Gender`: Customer gender
- `Age`: Customer age
- `Annual Income (k$)`: Annual income of the customer in thousand dollars
- `Spending Score (1-100)`: Score assigned by the mall based on customer spending behavior

## Features Used for Clustering

The K-Means model is trained using:

- `AnnualIncome`
- `SpendingScore`

These features are used because they directly represent customer purchasing power and spending behavior.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Methodology

1. Loaded the customer dataset.
2. Renamed columns for easier processing.
3. Checked dataset information, missing values, and basic statistics.
4. Performed exploratory data analysis.
5. Selected `AnnualIncome` and `SpendingScore` as clustering features.
6. Scaled the selected features using `StandardScaler`.
7. Used the Elbow Method and Silhouette Score to find the optimal number of clusters.
8. Trained the final K-Means model with `K=5`.
9. Visualized the final customer segments.

## Optimal Number of Clusters

The model was tested with different values of K from 2 to 10.

| K | Inertia | Silhouette Score |
|---|---------|------------------|
| 2 | 269.69 | 0.3213 |
| 3 | 157.70 | 0.4666 |
| 4 | 108.92 | 0.4939 |
| 5 | 65.57 | 0.5547 |
| 6 | 55.06 | 0.5399 |
| 7 | 44.86 | 0.5281 |
| 8 | 37.23 | 0.4552 |
| 9 | 32.39 | 0.4571 |
| 10 | 29.98 | 0.4432 |

`K=5` was selected because it achieved the best Silhouette Score of `0.5547`.

## Final Customer Segments

| Cluster | Segment Name | Customers | Avg Age | Avg Income | Avg Spending Score |
|---------|--------------|-----------|---------|------------|--------------------|
| 0 | Careful Spenders | 81 | 42.7 | $55.3k | 49.5 |
| 1 | High Value Targets | 39 | 32.7 | $86.5k | 82.1 |
| 2 | Careless Spenders | 22 | 25.3 | $25.7k | 79.4 |
| 3 | Standard Customers | 35 | 41.1 | $88.2k | 17.1 |
| 4 | Sensible Customers | 23 | 45.2 | $26.3k | 20.9 |

## Output Files

Running the script generates the following files:

- `eda_analysis.png`: Exploratory data analysis visualizations
- `elbow_silhouette.png`: Elbow Method and Silhouette Score plots
- `cluster_results.png`: Final customer cluster visualization
- `cluster_profiles.png`: Cluster profile comparison

## How to Run

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the Python script:

```bash
python customer_segmentation.py
```

Make sure `Mall_Customers.csv` is present in the same folder as `customer_segmentation.py`.

## Conclusion

The K-Means clustering model successfully grouped customers into five meaningful segments based on their annual income and spending score. These segments can help a retail store understand customer behavior and create targeted marketing strategies.
