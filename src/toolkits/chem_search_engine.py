import os

import requests

from .funcs import parallel_map


class ChemicalsDataSearchEngine:
    def __init__(self) -> None:
        # 国家危险化学品安全公共服务互联网平台
        self.home_url = "https://whpdj.mem.gov.cn"

        self.base_url = "https://whpdj.mem.gov.cn/internet/common/chemical"

        # 查询化学品列表
        self.get_chem_id_url = self.base_url + "/queryChemicalList"
        # 根据化学品唯一标识符查询化学品信息
        self.get_chem_info_url = self.base_url + "/queryChemicalById"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
            "Content-Type": "application/json",
        }

        self.file_dir = "/root/Documents/msds-qa/assets"

        self.no_msds_chemicals = []

    def get_idenDataId(self, chemName: str) -> str:
        """
        精准匹配获取化学品的唯一标识符

        :param chemName: 化学品名称

        :return: 化学品的唯一标识符
        """

        payload = {
            "status": "1",
            "chemName": str(chemName),
            "chemCas": "",
            "chemEnglishName": "",
            "isFuzzy": "0",
            "page": {"current": "1", "size": 10},
        }

        response = requests.post(
            self.get_chem_id_url,
            json=payload,
            headers=self.headers,
        )
        json_object = response.json()

        if json_object["obj"] is not None and json_object["obj"]["records"]:
            return json_object["obj"]["records"][0]["idenDataId"]
        else:
            return None

    def get_all_ChemNames(self) -> list[str]:
        """
        获取所有化学品的名称

        :return: 化学品的名称列表
        """

        payload = {
            "status": "1",
            "chemName": "",
            "chemCas": "",
            "chemEnglishName": "",
            "isFuzzy": "0",
            "page": {"current": "1", "size": 7173},
        }

        response = requests.post(
            self.get_chem_id_url,
            json=payload,
            headers=self.headers,
        )
        json_object = response.json()

        return parallel_map(
            lambda record: record["chemName"],
            json_object["obj"]["records"],
            max_workers=10,
            enable_tqdm=True,
        )

    def get_chemInfo(self, idenDataId: str) -> dict:
        """
        根据化学品的唯一标识符获取化学品信息

        :param idenDataId: 化学品的唯一标识符

        :return: 化学品信息
        """

        payload = {"idenDataId": idenDataId, "status": "1"}

        response = requests.post(
            self.get_chem_info_url,
            json=payload,
            headers=self.headers,
        )
        json_object = response.json()
        return json_object["obj"]

    def get_fileInfo(self, response: dict) -> dict:
        """
        获取化学品的安全文件名称

        :param response: 化学品信息的响应

        :return: 包含文件名称和下载地址的字典
        :rtype: dict
        """

        ret = {
            "safetyFileName": response["safetyFileName"],
            "safetyFileUrl": (
                self.home_url + response["safetyFileUrl"]
                if response["safetyFileUrl"]
                else None
            ),
        }
        return ret

    def download_msds_by_name(self, chemName: str) -> None:
        """
        下载化学品的安全数据表

        :param chemName: 化学品名称
        """

        idenDataId = self.get_idenDataId(chemName)

        cond1 = idenDataId is not None
        if cond1:
            chem_info = self.get_chemInfo(idenDataId)
            file_info = self.get_fileInfo(chem_info)

            cond2 = file_info["safetyFileUrl"] is not None

        if cond1 and cond2:
            file_name = file_info["safetyFileName"].split("@")[0]
            file_path = os.path.join(self.file_dir, f"{file_name}.pdf")

            if not os.path.exists(file_path):
                response = requests.get(
                    file_info["safetyFileUrl"],
                    headers=self.headers,
                    allow_redirects=True,
                )
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)

                print(f"Download Successfully")
        else:
            self.no_msds_chemicals.append(chemName)

    def download_all_msds(self) -> None:
        """
        下载化学品的安全数据表

        :param chemName: 化学品名称
        """

        chemNames = self.get_all_ChemNames()

        parallel_map(
            self.download_msds_by_name,
            chemNames,
            max_workers=10,
            enable_tqdm=True,
        )

        with open("/root/Documents/msds-qa/scripts/no_msds_chemicals.txt", "a") as f:
            for chem in self.no_msds_chemicals:
                f.write(f"{chem}\n")
        # for chemName in tqdm(chemNames):
        #     self.download_msds_by_name(chemName)

    def test_get_idenDataId(self) -> str:
        chemName = "氟化铵"
        idenDataId = self.get_idenDataId(chemName)
        return idenDataId

    def test_get_chemInfo(self) -> dict:
        idenDataId = "B2BFE2FE-225C-43F6-972D-32335472CAF9"
        chem_info = self.get_chemInfo(idenDataId)
        return chem_info

    def test_get_fileInfo(self) -> dict:
        idenDataId = "B2BFE2FE-225C-43F6-972D-32335472CAF9"
        chem_info = self.get_chemInfo(idenDataId)
        file_info = self.get_fileInfo(chem_info)
        return file_info

    def test_download_msds_by_name(self) -> None:
        chemName = "氟化铵"
        self.download_msds_by_name(chemName)


class ChemInfo:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # """
    # 化学品信息数据类
    # """

    # idenDataId: str
    # chemName: str
    # chemCas: str
    # chemAlias: str  # 化学品名称
    # chemEnglishName: str  # 英文名称
    # appearanceShape: str  # 外观与性状
    # ph: str  # pH值
    # meltPoint: str  # 熔点
    # boilPoint: str  # 沸点
    # relativeDensity: str  # 相对密度
    # relativeVaporDensity: str  # 相对蒸气密度
    # vaporPressure: str  # 蒸气压
    # combustionHeat: str  # 燃烧热
    # limitTemp: str  # 临界温度
    # limitPress: str  # 临界压力
    # octMatModulus: str  # 辛醇-水分配系数
    # flashPoint: str  # 闪点
    # autoIgnitionTemp: str  # 自燃温度
    # exploLowerLimit: str  # 爆炸下限
    # exploUpperLimit: str  # 爆炸上限
    # breakdownTemp: str  # 分解温度
    # viscosity: str  # 粘度
    # solubilty: str  # 溶解度
    # density: str  # 密度
    # specialDanger: str  # 燃烧与爆炸危险性
    # physcialChemDanger: str  # 活性反应
    # healthHazard: str  # 中毒表现
    # careerContactLimit: str  # 职业接触限值
    # environmentHazard: str  # 环境危害
    # firstMeasure: str  # 急救措施
    # leakageMeasure: str  # 泄漏应急措施
    # adviceProjectExtinguish: str  # 灭火方法
    # avoidMater: str  # 避免接触的物质
    # acuteToxicity: str  # 毒性
    # riskCategory: str  # 危险性类别
    # riskDesc: str  # 危险性说明
    # warnWord: str  # GHS警示词


if __name__ == "__main__":
    search_engine = ChemicalsDataSearchEngine()
    idenDataId = search_engine.test_get_idenDataId()
    chem_info = search_engine.get_chemInfo(idenDataId)
    file_info = search_engine.get_fileInfo(chem_info)

    print("化学品信息查询结果:")
    print(f"化学品的唯一标识符: {idenDataId}")
    print(f"化学品信息: {chem_info['chemName']}")
    print(f"安全文件名称: {file_info['safetyFileName']}")
    print(f"安全文件下载地址: {file_info['safetyFileUrl']}")
    # search_engine.download_all_msds()
