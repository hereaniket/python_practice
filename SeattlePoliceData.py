import pandas as pd

filePath = "/Users/aniketpathak/Documents/IMPORTANT/MS/CST-570/Topic-1/Backup.csv"

dataTypes = {
    "CAD Event Number": "int64",
    "Event Clearance Description": "string",
    "Call Type": "string",
    "Priority": "string",
    "Initial Call Type": "string",
    "Final Call Type": "string",
    "Original Time Queued": "string",
    "Arrived Time": "string",
    "Precinct": "string",
    "Sector": "string",
    "Beat": "string",
}

data = pd.read_csv(filepath_or_buffer=filePath, dtype=dataTypes)

sub_set = data[['CAD Event Number', 'Call Type', 'Original Time Queued']].head(1000)

# This will show how many different types of call type are there
print(sub_set["Call Type"].unique())

# Below will set the different call types to number 1 to 6
sub_set.loc[(sub_set['Call Type'] == '911'), 'Call Type'] = '1'
sub_set.loc[(sub_set['Call Type'] == 'TELEPHONE OTHER, NOT 911'), 'Call Type'] = '2'
sub_set.loc[(sub_set['Call Type'] == 'ONVIEW'), 'Call Type'] = '3'
sub_set.loc[(sub_set['Call Type'] == 'ALARM CALL (NOT POLICE ALARM)'), 'Call Type'] = '4'
sub_set.loc[(sub_set['Call Type'] == 'TEXT MESSAGE'), 'Call Type'] = '5'

print(sub_set["Call Type"].unique())

# This will print counts for all the call types
group_set = sub_set.groupby(['Call Type']).count()

print(group_set)

# This will plot the bar graph
group_set.plot.bar(rot=0)
