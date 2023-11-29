import pandas as pd
import numpy as np
import os

def split_records(dataFrame:pd.DataFrame) -> pd.DataFrame:
    outDataFrame = dataFrame
    for col in dataFrame.columns:
        if col.startswith("P"):
            dataFrame = dataFrame.applymap(lambda x: x.strip('()') if isinstance(x, str) else x)
            splited = dataFrame[col].str.split(",")
            splited = splited.apply(pd.Series)
            outDataFrame[col+"X"] = splited[0]
            outDataFrame[col+"Y"] = splited[1]
            del outDataFrame[col]
    return outDataFrame

def fuse_dataframes(path) -> pd.DataFrame:
    outDf = None
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                if outDf is None:
                    outDf = pd.read_csv(os.path.join(root, name),index_col=0)
                    print(outDf)
                else:
                    print(os.path.join(root, name))
                    np = os.path.join(root, name)
                    tmpDf = pd.read_csv(np,index_col=0)
                    outDf = pd.concat([outDf,tmpDf],ignore_index=True)
    return outDf



if __name__ == "__main__":
    df = fuse_dataframes("./HandContourRecognition/")
    #print(df)
    splited = split_records(df)
    print(splited)
    splited.to_csv("./HandContourRecognition/Dataset/unified.csv")
    print(splited)
