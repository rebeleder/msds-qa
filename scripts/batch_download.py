import pandas as pd

from src.toolkits import parallel_map
from src.toolkits.chem_search_engine import ChemicalsDataSearchEngine


def download_msds_for_chem(chem_name: str, search_engine: ChemicalsDataSearchEngine):
    search_engine.download_msds_by_name(chem_name)


if __name__ == "__main__":
    df = pd.read_csv("/root/Documents/msds-qa/scripts/chem_table.csv")
    search_engine = ChemicalsDataSearchEngine()
    parallel_map(
        lambda chem_name: download_msds_for_chem(chem_name, search_engine),
        df["危险品名称"],
        max_workers=10,
        enable_tqdm=True,
    )

    # 下载完成后，将没有下载到 MSDS 的化学品记录到文件中
    with open("/root/Documents/msds-qa/scripts/no_msds_chemicals.txt", "a") as f:
        for chem in search_engine.no_msds_chemicals:
            f.write(f"{chem}\n")
