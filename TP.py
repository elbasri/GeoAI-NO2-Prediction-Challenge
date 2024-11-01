import pandas as pd
from matplotlib import pyplot as plt
import itertools
import seaborn as sns

df = pd.read_csv("data/Train.csv")
print(df.head())

exlude_cols = ['Date', 'ID', 'ID_Zindi']

numeric_cols = df.select_dtypes(include=['number']).columns.difference(exlude_cols)
print(numeric_cols)

if 'Month' not in df.columns:
    df['Month'] = pd.to_datetime(df['Date']).dt.month

df[numeric_cols] = df[numeric_cols].fillna(df.groupby('Month')[numeric_cols].transform('mean'))
print("Data after NaN imputation by monthly mean:\n", df[numeric_cols].head())


#plt.scatter(df['Precipitation'], df['GT_NO2'], color='blue')
#plt.title('Precipation vs GT_NO2')
#plt.xlabel('Precipitation')
#plt.ylabel('GT_NO2')
#plt.show()

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

color_cycle = itertools.cycle(colors)

for col in numeric_cols:
    if col != 'GT_NO2':
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col], df['GT_NO2'], color=next(color_cycle), alpha=0.5)
        plt.title(f'{col} vs GT_NO2')
        plt.xlabel(col)
        plt.ylabel('GT_NO2')

plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Variables")


# ga3 les graphs/plots:
#plt.show()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['DayOfWeek'] = df['Date'].dt.dayofweek

df['is_weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
    
df['Season'] = df['Month'].apply(get_season)

# Display the DataFrame
print(df)

df['Precipitation_Category'] = pd.cut(df['Precipitation'], bins=[-1, 0.5, 2, df['Percipitation'].max()], labels=['Low', 'Medium', 'Hight'])

