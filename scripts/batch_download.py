from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.toolkits.chem_search_engine import chemicalsDataSearchEngine


def download_msds_for_chem(chem_name, search_engine):
    # 单个化学品的下载任务
    print(chem_name)
    search_engine.download_msds_by_name(chem_name)


if __name__ == "__main__":
    df = pd.read_csv("/root/Documents/msds-qa/scripts/chem_table.csv")
    search_engine = chemicalsDataSearchEngine()

    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 使用 tqdm 进度条
        list(
            tqdm(
                executor.map(
                    lambda chem_name: download_msds_for_chem(chem_name, search_engine),
                    df["危险品名称"],
                ),
                total=len(df),
                desc="Downloading",
                colour="green",
            )
        )

    # 下载完成后，将没有下载到 MSDS 的化学品记录到文件中
    with open("/root/Documents/msds-qa/scripts/no_msds_chemicals.txt", "a") as f:
        for chem in search_engine.no_msds_chemicals:
            f.write(f"{chem}\n")
