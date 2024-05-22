from os import listdir
from os.path import join
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    rootdir = "/media/mijun/Data/FluxNetExample/raw_data"
    output_dir = "/media/mijun/Data/FluxNetExample/processed_data"
    columns = ['TIMESTAMP_START', 'VPD_F', 'TA_F',
               'PA_F', 'WS_F', 'P_F', 'LW_IN_F', 'SW_IN_F',
               'LE_F_MDS', 'LE_F_MDS_QC',
               'LW_OUT', 'SW_OUT', 'G_F_MDS',
               'H_F_MDS']
    rng_colums = ['LW_IN_F', 'SW_IN_F', 'LW_OUT', 'SW_OUT', 'G_F_MDS']
    site_list = listdir(rootdir)
    site_list.sort()

    # get all sites
    sites = []
    for site in site_list:
        site_dir = join(rootdir, site)
        csv_list = listdir(site_dir)
        for csv in csv_list:
            if "FULLSET_HH" in csv:
                csv_path = join(site_dir, csv)
                sites.append(csv_path)
    print("get {} sites in total".format(len(sites)))
    for site_path in tqdm(sites):
        df = pd.read_csv(site_path)
        try:
            site = df[columns].copy()
        except Exception as e:
            print(e)
            continue
        site.loc[:, 'Date'] = site['TIMESTAMP_START'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d%H%M').date())
        dates = sorted(site['Date'].unique())
        site_data_list = []
        for date in dates:
            site_date = site[site['Date'] == date]
            if site_date.shape[0] != 48:
                # print("site data shape abnormal: {}".format(site_date.shape))
                continue
            rng_array = site_date[rng_colums].to_numpy()
            # if any value in rng_array is less than -9000, drop this date
            if (rng_array < -9000).any():
                continue

            correction_condition = (
                site_date["H_F_MDS"]/site_date["LE_F_MDS"]) >= 0
            rng = site_date['LW_IN_F'][correction_condition] + site_date['SW_IN_F'][correction_condition] - \
                site_date['LW_OUT'][correction_condition] - \
                site_date['SW_OUT'][correction_condition] - \
                site_date['G_F_MDS'][correction_condition]
            site_date.loc[correction_condition, "LE_F_MDS"] = rng / \
                (1+(site_date["H_F_MDS"][correction_condition] /
                 site_date["LE_F_MDS"][correction_condition]))
            site_data_list.append(site_date)
        if len(site_data_list) == 0:
            print("no site data in {}".format(site_path))
            continue
        site_data = pd.concat(site_data_list)
        site_name = site_path.split("/")[-1].split("_FLUXNET")[0]
        csv_name = "{}.csv".format(site_name)
        csv_path = join(output_dir, csv_name)
        site_data.to_csv(csv_path, index=False)
