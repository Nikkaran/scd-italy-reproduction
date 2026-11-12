import eurostat
import pandas as pd
from pathlib import Path

OUT = Path("data/italy_nuts3_panel.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Downloading GDP data...")
gdp_raw = eurostat.get_data_df('nama_10r_3gdp', flags=False)
gdp_raw.columns = [col.split('\\')[0] if '\\' in col else col for col in gdp_raw.columns]

# Just filter for Italy and let pandas grab the first available value per year
gdp = gdp_raw[gdp_raw['geo'].str.startswith('IT', na=False)].copy()

year_cols = [c for c in gdp.columns if str(c).isdigit()]
gdp_long = gdp.melt(id_vars=['geo'], value_vars=year_cols, var_name='year', value_name='gdp_pc')
gdp_long['year'] = gdp_long['year'].astype(int)
gdp_long = gdp_long[(gdp_long['year'] >= 2000) & (gdp_long['year'] <= 2022)].dropna(subset=['gdp_pc'])

# This line kills duplicates (like different industry types) by just taking the first row
gdp_long = gdp_long.groupby(['geo', 'year']).first().reset_index()

print("Downloading Employment data...")
emp_raw = eurostat.get_data_df('nama_10r_3empers', flags=False)
emp_raw.columns = [col.split('\\')[0] if '\\' in col else col for col in emp_raw.columns]
emp = emp_raw[emp_raw['geo'].str.startswith('IT', na=False)].copy()

emp_year_cols = [c for c in emp.columns if str(c).isdigit()]
emp_long = emp.melt(id_vars=['geo'], value_vars=emp_year_cols, var_name='year', value_name='employment')
emp_long['year'] = emp_long['year'].astype(int)
emp_long = emp_long[(emp_long['year'] >= 2000) & (emp_long['year'] <= 2022)].dropna(subset=['employment'])

# Kill duplicates here too
emp_long = emp_long.groupby(['geo', 'year']).first().reset_index()

print("Merging...")
panel = pd.merge(gdp_long, emp_long, on=['geo', 'year'], how='inner')
panel = panel.sort_values(['geo', 'year']).reset_index(drop=True)

print(f"\nSuccess! Final panel shape: {panel.shape}")
print(f"Provinces: {panel['geo'].nunique()}")
print(f"Sample codes: {panel['geo'].unique()[:5]}")

panel.to_parquet(OUT, index=False)
print(f"Saved to: {OUT}")