import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('seasonal_agriculture_performance_dataset.csv')
df.head()

print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSeason counts:")
print(df['Season'].value_counts())

df_clean = df.copy()
num_cols = df_clean.select_dtypes(include=np.number).columns
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Remaining missing values:", df_clean.isnull().sum().sum())

seasonal = df_clean.groupby('Season').agg(
    Farms=('Farm_ID','count'),
    Avg_Yield=('Yield_Tonnes_Ha','mean'),
    Median_Yield=('Yield_Tonnes_Ha','median'),
    Avg_Profit=('Profit_INR','mean'),
    Median_Profit=('Profit_INR','median'),
    Avg_Revenue=('Revenue_INR','mean'),
    Avg_Rainfall=('Rainfall_mm','mean'),
    Avg_Temp=('Avg_Temperature_C','mean'),
    Avg_Water=('Water_Used_m3','mean'),
    Water_Efficiency=('Water_Efficiency_t_per_1000m3','mean'),
    Disease_Risk=('Disease_Pest_Risk_pct','mean')
).round(2)
print(seasonal)

seasonal[['Avg_Yield']].plot(kind='bar', legend=False, figsize=(7,4))
plt.title('Average Yield by Season')
plt.ylabel('Tonnes per hectare')
plt.xticks(rotation=0)
plt.show()

seasonal[['Avg_Profit']].plot(kind='bar', legend=False, figsize=(7,4))
plt.title('Average Profit by Season')
plt.ylabel('Profit (INR)')
plt.xticks(rotation=0)
plt.axhline(0, color='black', linewidth=1)
plt.show()

seasonal[['Avg_Rainfall','Avg_Water']].plot(kind='bar', figsize=(8,4))
plt.title('Seasonal Rainfall and Water Usage')
plt.xticks(rotation=0)
plt.show()

seasonal[['Disease_Risk']].plot(kind='bar', legend=False, figsize=(7,4))
plt.title('Average Disease & Pest Risk by Season')
plt.ylabel('Risk (%)')
plt.xticks(rotation=0)
plt.show()

crop_season = df_clean.groupby(['Crop','Season'])['Yield_Tonnes_Ha'].mean().unstack()
print(crop_season.round(2))

crop_season.plot(kind='bar', figsize=(10,5))
plt.title('Average Yield by Crop and Season')
plt.ylabel('Yield (Tonnes/Ha)')
plt.xticks(rotation=45)
plt.show()

cols = ['Yield_Tonnes_Ha','Rainfall_mm','Avg_Temperature_C',
        'Humidity_pct','Sunlight_Hours_Day','Soil_Moisture_pct',
        'Fertilizer_kg_ha','Pesticide_Litre_ha','Seed_Quality_Score',
        'Water_Used_m3','Disease_Pest_Risk_pct','Profit_INR']

yield_corr = df_clean[cols].corr(numeric_only=True)['Yield_Tonnes_Ha'].sort_values(ascending=False)
print(yield_corr)

