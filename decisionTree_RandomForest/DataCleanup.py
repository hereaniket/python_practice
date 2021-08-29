import pandas as pd

filepath = "../data/customer_churn/Data.csv"
data_type = {
    "customer_id": "string",
    "gender": "string",
    "senior_citizen": "string",
    "partner": "string",
    "dependents": "string",
    "tenure": "string",
    "phone_service": "string",
    "multiple_lines": "string",
    "internet_service": "string",
    "online_security": "string",
    "online_backup": "string",
    "device_protection": "string",
    "tech_support": "string",
    "streaming_service": "string",
    "streaming_movies": "string",
    "contract": "string",
    "paperless_billing": "string",
    "payment_method": "string",
    "monthly_charges": "string",
    "total_charges": "string",
    "churn": "string"
}

cust_data = pd.read_csv(filepath_or_buffer=filepath, dtype=data_type)

cust_data.loc[(cust_data['multiple_lines'] == 'No phone service'), 'multiple_lines'] = '0'
cust_data.loc[(cust_data['multiple_lines'] == 'No'), 'multiple_lines'] = '0'
cust_data.loc[(cust_data['multiple_lines'] == 'Yes'), 'multiple_lines'] = '1'

cust_data.loc[(cust_data['online_backup'] == 'No internet service'), 'online_backup'] = '0'
cust_data.loc[(cust_data['online_backup'] == 'No'), 'online_backup'] = '0'
cust_data.loc[(cust_data['online_backup'] == 'Yes'), 'online_backup'] = '1'

cust_data.loc[(cust_data['online_security'] == 'No internet service'), 'online_security'] = '0'
cust_data.loc[(cust_data['online_security'] == 'No'), 'online_security'] = '0'
cust_data.loc[(cust_data['online_security'] == 'Yes'), 'online_security'] = '1'

cust_data.loc[(cust_data['device_protection'] == 'No internet service'), 'device_protection'] = '0'
cust_data.loc[(cust_data['device_protection'] == 'No'), 'device_protection'] = '0'
cust_data.loc[(cust_data['device_protection'] == 'Yes'), 'device_protection'] = '1'

cust_data.loc[(cust_data['tech_support'] == 'No internet service'), 'tech_support'] = '0'
cust_data.loc[(cust_data['tech_support'] == 'No'), 'tech_support'] = '0'
cust_data.loc[(cust_data['tech_support'] == 'Yes'), 'tech_support'] = '1'

cust_data.loc[(cust_data['streaming_service'] == 'No internet service'), 'streaming_service'] = '0'
cust_data.loc[(cust_data['streaming_service'] == 'No'), 'streaming_service'] = '0'
cust_data.loc[(cust_data['streaming_service'] == 'Yes'), 'streaming_service'] = '1'

cust_data.loc[(cust_data['streaming_movies'] == 'No internet service'), 'streaming_movies'] = '0'
cust_data.loc[(cust_data['streaming_movies'] == 'No'), 'streaming_movies'] = '0'
cust_data.loc[(cust_data['streaming_movies'] == 'Yes'), 'streaming_movies'] = '1'

cust_data.loc[(cust_data['churn'] == 'No'), 'churn'] = '0'
cust_data.loc[(cust_data['churn'] == 'Yes'), 'churn'] = '1'

cust_data.loc[(cust_data['paperless_billing'] == 'No'), 'paperless_billing'] = '0'
cust_data.loc[(cust_data['paperless_billing'] == 'Yes'), 'paperless_billing'] = '1'

cust_data.loc[(cust_data['phone_service'] == 'No'), 'phone_service'] = '0'
cust_data.loc[(cust_data['phone_service'] == 'Yes'), 'phone_service'] = '1'

cust_data.loc[(cust_data['dependents'] == 'No'), 'dependents'] = '0'
cust_data.loc[(cust_data['dependents'] == 'Yes'), 'dependents'] = '1'

cust_data.loc[(cust_data['partner'] == 'No'), 'partner'] = '0'
cust_data.loc[(cust_data['partner'] == 'Yes'), 'partner'] = '1'

cust_data.loc[(cust_data['total_charges'] == ' '), 'total_charges'] = '0'
cust_data.loc[(cust_data['total_charges'] == ''), 'total_charges'] = '0'

cust_data.to_csv("data/Data_Formatted.csv")

print(cust_data.info())
