import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# load file
filepath = "SwabData_SFE_total_copynumb_region.csv"
region_file = pd.read_csv(filepath)

# format date + CT
region_file['collection_date'] = pd.to_datetime(region_file['collection_date'], errors='coerce')
region_file['CT'] = pd.to_numeric(region_file['CT'], errors='coerce')

# filter for swab samples only
region_file = region_file[region_file['type'] == "S"].copy()

# tag positives
region_file['is_positive'] = region_file['CT'] > 0

# extract year
region_file['year'] = region_file['collection_date'].dt.year

# get total number of tests per year
total_2019 = len(region_file[region_file['year'] == 2019])
total_2020 = len(region_file[region_file['year'] == 2020])
total_tests = {2019: total_2019, 2020: total_2020}

print(total_2019)

# align dates
region_file['aligned_date'] = region_file['collection_date'].apply(lambda d: d.replace(year=2020))

# remove 2021
region_file = region_file[region_file['year'] != 2021]

# store results
results = []

# loop through regions
for region in range(1, 5):
    region_df = region_file[region_file['region'] == region].copy()

    # loop through years
    for year in sorted(region_df['year'].unique()):
        year_data = region_df[region_df['year'] == year].copy()

        # set total based on year
        year_total = total_tests[year]

        # set running total
        cumulative_pos = 0

        # loop through dates
        for date in sorted(year_data['collection_date'].unique()):
            date_data = year_data[year_data['collection_date'] == date]

            # update total
            cumulative_pos += date_data['is_positive'].sum()

            # get prop
            proportion = cumulative_pos / year_total if year_total > 0 else 0

            # store data
            results.append({
                'region': region,
                'year': year,
                'date': date,
                'aligned_date': pd.to_datetime(date).replace(year=2020),
                'cumulative_total': year_total,
                'cumulative_positive': cumulative_pos,
                'proportion_infected': proportion
            })

# to dataframe
result_df = pd.DataFrame(results)

# plot
plt.figure(figsize=(14, 7))

# line colors for regions
region_colors = {
    1: '#FB0650',
    2: '#1E88E5',
    3: '#07FF81',
    4: '#004D40'
}

# line styles for years
year_styles = {
    2019: 'solid',
    2020: 'dashed'
}

# plot all
for region in sorted(result_df['region'].unique()):
    for year in sorted(result_df[result_df['region'] == region]['year'].unique()):
        plot_data = result_df[(result_df['region'] == region) & (result_df['year'] == year)]
        plt.plot(
            plot_data['aligned_date'],
            plot_data['proportion_infected'],
            label=f'Region {region}, {year}',
            color=region_colors[region],
            linestyle=year_styles.get(year, 'solid'),  # fallback if year not mapped
            linewidth=2
        )

# format plot
#plt.title('Figure 4. Cumulative Proportion Infected by Year and Region')
plt.xlabel('Time')
plt.ylabel('Cumulative Proportion Infected')
plt.ylim(0, 0.3)
plt.grid(True, linestyle='--', alpha=0.7)

# create custom legend for region colors
region_legend = [
    Line2D([0], [0], color=color, lw=3, linestyle='solid', label=f'Region {r}')
    for r, color in region_colors.items()
]

# create custom legend for year line styles
year_legend = [
    Line2D([0], [0], color='black', lw=2, linestyle=style, label=str(year))
    for year, style in year_styles.items()
]

# add both legends
first_legend = plt.legend(handles=region_legend, title='Region', loc='upper left')
plt.gca().add_artist(first_legend)  # make room for second legend

plt.legend(handles=year_legend, title='Year', loc='upper right')

# month formatting
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout()
plt.show()
