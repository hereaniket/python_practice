import pandas as pd

filepath = "data/stateandyearwise.csv"
dataTypes = {
    "state_name": "string",
    "2010": "string",
    "2011": "string",
    "2012": "string",
    "2013": "string",
    "2014": "string",
    "2015": "string",
    "2016": "string",
    "2017": "string",
    "2018": "string",
    "2019": "string",
}
raw_data = pd.read_csv(filepath_or_buffer=filepath, dtype=dataTypes)

frames = {}
for i in range(0, raw_data['state_name'].count()):
    data = {
        'state_name': [raw_data['state_name'][i], raw_data['state_name'][i], raw_data['state_name'][i],
                       raw_data['state_name'][i], raw_data['state_name'][i], raw_data['state_name'][i],
                       raw_data['state_name'][i], raw_data['state_name'][i], raw_data['state_name'][i],
                       raw_data['state_name'][i]],
        'year': ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'],
        'population': [raw_data['2010'][i], raw_data['2011'][i], raw_data['2012'][i], raw_data['2013'][i], raw_data['2014'][i],
                       raw_data['2015'][i], raw_data['2016'][i], raw_data['2017'][i], raw_data['2018'][i], raw_data['2019'][i]]
    }
    dt = pd.DataFrame(data)
    frames[i] = dt

new_population_df = pd.concat(frames)
print(new_population_df.head())
new_population_df.to_csv('data/out.csv', index=False)


# filtering data
filteredData = new_population_df.where(new_population_df["state_name"] == "Alabama", inplace=True)
print(filteredData)
