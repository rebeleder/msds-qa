import pandas as pd

from src.db import Neo4jDB
from src.model import SiliconflowClient
from src.toolkits import ChemicalsDataSearchEngine

chat_model = SiliconflowClient().get_chat_model()
embed_model = SiliconflowClient().get_embed_model()

db = Neo4jDB(chat_model=chat_model, embed_model=embed_model)
engine = ChemicalsDataSearchEngine()

all_name = engine.get_all_ChemNames()


idenDataId_list = []
chemName_list = []
chemCas_list = []
chemAlias_list = []
chemEnglishName_list = []
appearanceShape_list = []
ph_list = []
meltPoint_list = []
boilPoint_list = []
relativeDensity_list = []
relativeVaporDensity_list = []
vaporPressure_list = []
combustionHeat_list = []
limitTemp_list = []
limitPress_list = []
octMatModulus_list = []
flashPoint_list = []
autoIgnitionTemp_list = []
exploLowerLimit_list = []
exploUpperLimit_list = []
breakdownTemp_list = []
viscosity_list = []
solubilty_list = []
density_list = []
specialDanger_list = []
physcialChemDanger_list = []
healthHazard_list = []
careerContactLimit_list = []
environmentHazard_list = []
firstMeasure_list = []
leakageMeasure_list = []
adviceProjectExtinguish_list = []
avoidMater_list = []
acuteToxicity_list = []
riskCategory_list = []
riskDesc_list = []
warnWord_list = []

all_name = engine.get_all_ChemNames()

for chem in all_name:
    try:
        idenDataId = engine.get_idenDataId(chem)
        if idenDataId:
            chem_info = engine.get_chemInfo(idenDataId)

            # 将化学品属性添加到对应的列表中
            idenDataId_list.append(chem_info.get("idenDataId", ""))
            chemName_list.append(chem_info.get("chemName", ""))
            chemCas_list.append(chem_info.get("chemCas", ""))
            chemAlias_list.append(chem_info.get("chemAlias", "暂无别名"))
            chemEnglishName_list.append(
                chem_info.get("chemEnglishName", "暂无英文名称")
            )
            appearanceShape_list.append(chem_info.get("appearanceShape", ""))
            ph_list.append(chem_info.get("ph", ""))
            meltPoint_list.append(chem_info.get("meltPoint", ""))
            boilPoint_list.append(chem_info.get("boilPoint", ""))
            relativeDensity_list.append(chem_info.get("relativeDensity", ""))
            relativeVaporDensity_list.append(chem_info.get("relativeVaporDensity", ""))
            vaporPressure_list.append(chem_info.get("vaporPressure", ""))
            combustionHeat_list.append(chem_info.get("combustionHeat", ""))
            limitTemp_list.append(chem_info.get("limitTemp", ""))
            limitPress_list.append(chem_info.get("limitPress", ""))
            octMatModulus_list.append(chem_info.get("octMatModulus", ""))
            flashPoint_list.append(chem_info.get("flashPoint", ""))
            autoIgnitionTemp_list.append(chem_info.get("autoIgnitionTemp", ""))
            exploLowerLimit_list.append(chem_info.get("exploLowerLimit", ""))
            exploUpperLimit_list.append(chem_info.get("exploUpperLimit", ""))
            breakdownTemp_list.append(chem_info.get("breakdownTemp", ""))
            viscosity_list.append(chem_info.get("viscosity", ""))
            solubilty_list.append(chem_info.get("solubilty", ""))
            density_list.append(chem_info.get("density", ""))
            specialDanger_list.append(chem_info.get("specialDanger", ""))
            physcialChemDanger_list.append(chem_info.get("physcialChemDanger", ""))
            healthHazard_list.append(chem_info.get("healthHazard", ""))
            careerContactLimit_list.append(chem_info.get("careerContactLimit", ""))
            environmentHazard_list.append(chem_info.get("environmentHazard", ""))
            firstMeasure_list.append(chem_info.get("firstMeasure", ""))
            leakageMeasure_list.append(chem_info.get("leakageMeasure", ""))
            adviceProjectExtinguish_list.append(
                chem_info.get("adviceProjectExtinguish", "")
            )
            avoidMater_list.append(chem_info.get("avoidMater", ""))
            acuteToxicity_list.append(chem_info.get("acuteToxicity", ""))
            riskCategory_list.append(chem_info.get("riskCategory", ""))
            riskDesc_list.append(chem_info.get("riskDesc", ""))
            warnWord_list.append(chem_info.get("warnWord", ""))
            pd.DataFrame(
                {
                    "idenDataId": idenDataId_list,
                    "chemName": chemName_list,
                    "chemCas": chemCas_list,
                    "chemAlias": chemAlias_list,
                    "chemEnglishName": chemEnglishName_list,
                    "appearanceShape": appearanceShape_list,
                    "ph": ph_list,
                    "meltPoint": meltPoint_list,
                    "boilPoint": boilPoint_list,
                    "relativeDensity": relativeDensity_list,
                    "relativeVaporDensity": relativeVaporDensity_list,
                    "vaporPressure": vaporPressure_list,
                    "combustionHeat": combustionHeat_list,
                    "limitTemp": limitTemp_list,
                    "limitPress": limitPress_list,
                    "octMatModulus": octMatModulus_list,
                    "flashPoint": flashPoint_list,
                    "autoIgnitionTemp": autoIgnitionTemp_list,
                    "exploLowerLimit": exploLowerLimit_list,
                    "exploUpperLimit": exploUpperLimit_list,
                    "breakdownTemp": breakdownTemp_list,
                    "viscosity": viscosity_list,
                    "solubilty": solubilty_list,
                    "density": density_list,
                    "specialDanger": specialDanger_list,
                    "physcialChemDanger": physcialChemDanger_list,
                    "healthHazard": healthHazard_list,
                    "careerContactLimit": careerContactLimit_list,
                    "environmentHazard": environmentHazard_list,
                    "firstMeasure": firstMeasure_list,
                    "leakageMeasure": leakageMeasure_list,
                    "adviceProjectExtinguish": adviceProjectExtinguish_list,
                    "avoidMater": avoidMater_list,
                    "acuteToxicity": acuteToxicity_list,
                    "riskCategory": riskCategory_list,
                    "riskDesc": riskDesc_list,
                    "warnWord": warnWord_list,
                }
            ).to_json(
                "chemicals_info.json", orient="records", force_ascii=False, indent=4
            )

        else:
            print(f"No idenDataId found for: {chem}")
    except Exception as e:
        print(f"Error processing {chem}: {e}")
        continue
