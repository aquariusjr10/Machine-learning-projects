import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

df = pd.read_csv("Dataset/predictive_maintenance_dataset.csv")

if df.empty:
    print("Dataset is empty")
else:
    print("Dataset loaded successfully")

# EDA
df.head()
df.info()
df.describe()
print("Number of null values:", df.isnull().sum())
print("Duplicates rows:", df.duplicated().sum())

# Data Cleaning
df.dropna(inplace=True)
print("Number of rows that have failure:", df[df["failure"] == 1].shape[0])

print("Dropping duplicates...")
df.drop_duplicates(inplace=True)
print("Duplicates rows after dropping:", df.duplicated().sum())


df["device_model"] = df["device"].apply(lambda x: x[:4])
df["device_model"].value_counts()


# Create a figure with 1 row and 2 columns
plt.figure(figsize=(12, 6))
# Plot the distribution of 'failure' with respect to 'device' for failure=1
plt.subplot(1, 2, 1)
sns.countplot(x="device_model", data=df.loc[df["failure"] == 1])
plt.title("Distribution of Failure (failure=1) with respect to Device")

# Plot the distribution of 'failure' with respect to 'device' for failure=0
plt.subplot(1, 2, 2)
sns.countplot(x="device_model", data=df.loc[df["failure"] == 0])
plt.title("Distribution of Failure (failure=0) with respect to Device")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


# Create histograms to visualize the distribution of selected metrics with 'failure' as hue
plt.figure(figsize=(4 * 5, 2 * 5))
print("Distribution for failure is 0")
mask = df.failure > 0
for i, col in enumerate(
    [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric9",
    ]
):
    plt.subplot(2, 4, i + 1)
    sns.histplot(data=df.loc[mask], x=col, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Convert the 'date' column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract and format the 'month' column for plotting
df["month"] = df["date"].dt.to_period("M")
df["month"] = df["month"].dt.strftime("%Y-%m")

# Create a line plot to visualize 'failure' over time by month
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="month", y="failure")
plt.xticks(rotation=45)
plt.title("Failure over Time by Month")

# Select only numeric columns for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
correlation_matrix = numeric_cols.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
