## This code was built by Arnor Elvarsson to prepare a combined OD matrix for the canton of Bern, Switzerland. 
## It reads in two CSV files containing traffic flow data for public transport (oev) and private motorized transport (miv), 
## reshapes the data, merges it with zone names, and outputs the final combined OD matrix to an Excel file.



import pandas as pd
import numpy as np

df1 = pd.read_csv(r'C:/Users/spadmin/Documents/infraScan/infraScanRail/data/Traffic_Flow/OD/Bern/BE_2019_OV_Ist2019_DWV.csv')
tmp1 = df1.melt(id_vars=['code'])
tmp1['verkehrsmittel'] = "oev"
#tmp1['ziel_code'] = df1.melt(col_level=0).dropna()[0].to_numpy()
tmp1 = tmp1.rename({'code':'quelle_code','variable':'ziel_code', 'value': 'wert'}, axis=1)
df1 = tmp1

df2 = pd.read_csv(r'C:/Users/spadmin/Documents/infraScan/infraScanRail/data/Traffic_Flow/OD/Bern/BE_2019_MIV_Ist2019_DWV.csv', header=None)#, index_col=0)
df2.columns = pd.MultiIndex.from_arrays([df2.iloc[0], df2.iloc[1]])
df2 = df2.iloc[2:]
tmp2 = df2.melt(id_vars=['code'], col_level=1)
tmp2['ziel_code'] = df2.melt(col_level=0).dropna()[0].to_numpy()
tmp2['verkehrsmittel'] = "miv"
tmp2 = tmp2.rename({'code':'quelle_code', 1: 'class', 'value': 'wert'}, axis=1)
df2 = tmp2[tmp2['class'] == 'Nr. 1']
df2 = df2.drop('class', axis=1)

merged_df = pd.concat([df1, df2], ignore_index=True)

df_names = pd.read_csv(r'C:/Users/spadmin/Documents/infraScan/infraScanRail/data/Traffic_Flow/OD/Bern/ZonenID-Name.csv', index_col=0)
merged_df = merged_df.set_index('quelle_code')
tmp =merged_df.join(df_names, how='inner', on='quelle_code')#, lsuffix="_caller", rsuffix="_other")
tmp["ziel_code"] = tmp["ziel_code"].astype(np.int64)
tmp= tmp.rename({'name':'quelle_name'}, axis=1)
tmp = tmp.reset_index()
#tmp = tmp.set_index('ziel_code')

df_names.index.names = ['ziel_code']
df_names = df_names.reset_index()
tmp =tmp.merge(df_names, how='inner', left_on='ziel_code',right_on='ziel_code')#, lsuffix="_caller", rsuffix="_other")
#tmp = pd.merge(tmp, df_names, how='inner', on='ziel_code')#.reindex(df_names.index)
df_names.index.names = ['code']
tmp= tmp.rename({'name':'ziel_name'}, axis=1)



output_order = [
    "jahr",
    "zeithorizonttyp",
    "quelle_gebietart",
    "quelle_code",
    "quelle_name",
    "ziel_gebietart",
    "ziel_code",
    "ziel_name",
    "verkehrsmittel",
    "wert",
    "einheit",
    "gebietsstand"
]

tmp['jahr'] = 2019
tmp['zeithorizonttyp'] = "modellierter Ist-Zustand"
tmp['quelle_gebietart'] = "Gemeinde"
tmp['ziel_gebietart'] = "Gemeinde"
tmp['einheit'] = "Personenwege an einem durchschnittlichen Werktag (DWV)"
tmp['gebietsstand'] = "2019"

output_df = tmp[output_order]
#output_df.to_excel(r"C:/Users/spadmin/Documents/infraScan/infraScanRail/data/Traffic_Flow/OD/Bern/BE_2019_Ist2019_DWV.xlsx", index=False)
output_df.to_csv(r"C:/Users/spadmin/Documents/infraScan/infraScanRail/data/Traffic_Flow/OD/Bern/BE_2019_Ist2019_DWV.csv", sep=';', encoding='utf-8', index=False)
