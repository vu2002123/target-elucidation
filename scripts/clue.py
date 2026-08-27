import os
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.write_gctx as wg

gctoo = parse("/home/vu2002123/target-elucidation/data/interim/PCP.gct")

WORK_DIR = "/home/vu2002123/target-elucidation/data/interim/"
os.chdir(WORK_DIR)
wg.write(gctoo, "PCP")
