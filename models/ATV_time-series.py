import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "SwabData_SFE_total_copynumb_region.csv"
df = pd.read_csv(file_path)

# Convert 'collection_date' to datetime and 'CT' to numeric
df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
df['CT'] = pd.to_numeric(df['CT'], errors='coerce')

# filter for sample type 's'
s_samples = df[df['type'] == "S"].copy()

# Filter for positive infections (CT > 0)
s_samples['is_positive'] = s_samples['CT'] > 0

# Extract year and normalize dates to the same reference year (e.g., 2020)
s_samples['year'] = s_samples['collection_date'].dt.year

# get total number of tests per year
total_2019 = len(s_samples[s_samples['year'] == 2019])
total_2020 = len(s_samples[s_samples['year'] == 2020])
total_tests = {2019: total_2019, 2020: total_2020, 2021: 1}

print(total_2019)
print(total_2020)

s_samples['month_day'] = s_samples['collection_date'].dt.strftime('%m-%d')
s_samples['aligned_date'] = s_samples['collection_date'].apply(lambda d: d.replace(year=2020))

# sort data within each year
s_samples = s_samples.sort_values('collection_date')

# calculate running counts by year
results = []

# process each year separately
for year in sorted(s_samples['year'].unique()):
    year_data = s_samples[s_samples['year'] == year].copy()

    # calculate cumulative counts for this year
    cumulative_total = total_tests[year]
    cumulative_pos = 0

    for date in sorted(year_data['collection_date'].unique()):
        date_data = year_data[year_data['collection_date'] == date]

        # add to running totals
        date_count = len(date_data)
        date_pos = date_data['is_positive'].sum()

        cumulative_pos += date_pos

        # calculate proportion
        proportion = cumulative_pos / cumulative_total if cumulative_total > 0 else 0

        # store result
        results.append({
            'year': year,
            'date': date,
            'aligned_date': pd.to_datetime(date).replace(year=2020),
            'cumulative_total': cumulative_total,
            'cumulative_positive': cumulative_pos,
            'proportion_infected': proportion
        })

# convert to DF
result_df = pd.DataFrame(results)

# remove 2021
result_df = result_df[result_df['year'] != 2021]

# plot running proportion by year
plt.figure(figsize=(12, 6))

years = sorted(result_df['year'].unique())
colors = ['#FB0650', '#1E88E5']

for i, year in enumerate(years):
    year_data = result_df[result_df['year'] == year]
    plt.plot(
        year_data['aligned_date'],
        year_data['proportion_infected'],
        marker='o',
        linestyle='-',
        color=colors[i % len(colors)],
        label=f'{year}'
    )

#plt.title('Figure 3. Cumulative Proportion Infected by Year')
plt.xlabel('Time')
plt.ylabel('Cumulative Proportion Infected')

#plt.ylim(0, 0.4)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Year')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
plt.tight_layout()
plt.show()

# stats
print("Final statistics by year:")
final_stats = result_df.groupby('year').last()[['cumulative_total', 'cumulative_positive', 'proportion_infected']]
print(final_stats)
