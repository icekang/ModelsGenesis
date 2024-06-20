# CF_PRES
import yaml
import sys
sys.path.append('../pytorch')
from utils import KFoldNNUNetTabularDataModule
import pandas as pd
from pathlib import Path


def save_to_jepa_data(df, path):
    data_directory = Path('/home/gridsan/nchutisilp/datasets/Unlabeled_OCT_by_CADx/Unlabeled_OCT_by_CADx/')

    mp4_classification_df = df.copy()
    mp4_classification_df = mp4_classification_df.drop(columns=['USUBJID'])
    mp4_classification_df[f'{inputName}_image_path'] = mp4_classification_df[f'{inputName}_image_path'].apply(lambda x: str(data_directory / 'MP4' / f'{x}.mp4'))

    mp4_classification_df.to_csv(path, index=False, sep=' ', columns=mp4_classification_df.columns[::-1], header=False)


with open('/home/gridsan/nchutisilp/projects/ModelsGenesis/pytorch/configs/fine_tune_config-regression-full-3d_32x160x128.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for fold in range(3):
    config['data']['fold'] = fold

    config['data']['input_modality'] = 'final'
    config['data']['output_modality'] = 'final'
    config['data']['output_metrics'] = ['CF_PRES']
    config['data']['tabular_data_directory'] = '/home/gridsan/nchutisilp/projects/ModelsGenesis/notebooks/tabular_data'

    # Data
    dm = KFoldNNUNetTabularDataModule(config=config)

    # create subject from the CSV!
    outputModalityDf = pd.read_csv(dm.modalityToDataframePath[dm.outputModality])
    inputName = dm.modalityToName[dm.inputModality]
    outputModalityDf = outputModalityDf[['USUBJID'] + dm.outputMetrics + [f'{inputName}_image_path']]
    outputModalityDf.dropna(inplace=True)

    stage = 'fit' # 'fit' or 'test'
    train_subject_ids, val_subject_ids = dm._getSplit(stage, outputModalityDf=outputModalityDf)
    _trainDF = outputModalityDf[outputModalityDf['USUBJID'].isin(train_subject_ids)]
    _valDF = outputModalityDf[outputModalityDf['USUBJID'].isin(val_subject_ids)]

    ids = _trainDF['USUBJID']
    assert len(ids) == len(set(ids)), 'Train IDs are not unique'
    ids = _valDF['USUBJID']
    assert len(ids) == len(set(ids)), 'Val ID are not unique'

    _train_positive_rate = 100 * len(_trainDF[_trainDF['CF_PRES'] == 1]) / len(_trainDF)
    _val_positive_rate = 100 * len(_valDF[_valDF['CF_PRES'] == 1]) / len(_valDF)
    print(f'Fold {fold} {dm.fold}')
    print(f'Train positive rate: {_train_positive_rate:.2f}%')
    print(f'Val positive rate: {_val_positive_rate:.2f}%')
    save_to_jepa_data(_trainDF, f'tabular_data/cf_pres_train_fold{fold}.csv')
    save_to_jepa_data(_valDF, f'tabular_data/cf_pres_val_fold{fold}.csv')
    print(f'save to tabular_data/cf_pres_train_fold{fold}.csv')
    print(f'save to tabular_data/cf_pres_val_fold{fold}.csv')

stage = 'test'
test_subject_ids, _ = dm._getSplit(stage, outputModalityDf=outputModalityDf)
_testDf = outputModalityDf[outputModalityDf['USUBJID'].isin(test_subject_ids)]
ids = _testDf['USUBJID'].tolist()
assert len(ids) == len(set(ids)), 'Test ID are not unique'

_train_positive_rate = 100 * len(_testDf[_testDf['CF_PRES'] == 1]) / len(_testDf)
print(f'Test positive rate: {_train_positive_rate:.2f}%')
save_to_jepa_data(_testDf, f'tabular_data/cf_pres_test.csv')