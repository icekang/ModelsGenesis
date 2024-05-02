# # We need to explore the encoded vector of the nnUNet to create an decoder to regress / classify the output instead of segmentation.

from pathlib import Path
import pandas as pd
import re
import unidecode
import numpy as np

unlabeled_data_dir = Path('/storage_bizon/naravich/Unlabeled_OCT_by_CADx/') # Yiqing filtered data

for excel_file_name, Pre_or_Post in zip(['T1A_PRE_QUANT_LESION.xlsx', 'T5A_POST_QUANT_LESION.xlsx'], ['Pre', 'Post']):
    pre_or_post_ivl_df = pd.read_excel(f'tabular_data/{excel_file_name}', skiprows=4)
    pre_or_post_ivl_df.head()

    # ### Rename columns to their code names `FLAG_CAL = Flag for calcified nodules` -> `FLAG_CAL`

    column_names = pre_or_post_ivl_df.columns
    column_names = {column_name:column_name.split(' = ')[0] for column_name in column_names}
    column_names
    pre_or_post_ivl_df.rename(columns=column_names, inplace=True)
    pre_or_post_ivl_df.head()

    # ### Extract the Unique Subject ID (USUBJID) to the format we use in the dataset `CP 61774-105-001` -> `105-001`
    # That is [A-Z]{2} [0-9]{5}-[0-9]{3}-[0-3]{3} -> [0-9]{3}-[0-9]{3}



    def format_subject_id(subject_id: str):
        # Define the pattern
        pattern = r"[A-Z]{2} [0-9]{4,5}-([0-9]{3}-[0-9]{3})"

        # Compile the pattern
        regex = re.compile(pattern)

        # Search for the pattern in the input string
        match = regex.search(unidecode.unidecode(subject_id))

        # If a match is found, extract the desired part using the replacement pattern
        if match:
            extracted_part = re.sub(r".*?([0-9]{3}-[0-9]{3})", r"\1", match.group(1))
            return extracted_part
        raise ValueError(f"No match found {subject_id.strip().replace(u'\xa0', ' ')}")

    pre_or_post_ivl_df['USUBJID'].apply(lambda x: format_subject_id(x))

    pre_or_post_ivl_df['USUBJID'] = pre_or_post_ivl_df['USUBJID'].apply(lambda x: format_subject_id(x))
    pre_or_post_ivl_df.head()

    # ### Select only the columns we are interested in
    # 
    # ```
    # MLAS_AS = Area stenosis (%)	
    # MLAS_SCA = Superficial calcium arc (°)	
    # MLAS_MSCT = Maximum superficial calcium thickness (mm)	
    # MCS_LA = Maximum calcium site -  Lumen area (mm2)	
    # MCS_AS = Maximum calcium site - Area stenosis (%)	
    # MCS_SCA = Maximum calcium site -  Superficial calcium arc (°)	
    # MCS_MSCT = Maximum calcium site -   Maximum superficial calcium thickness (mm)	
    # MCCS_LA = Maximum continuous calcium site - Lumen area (mm2)	
    # MCCS_AS = Maximum continuous calcium site - Area stenosis (%)	
    # MCCS_SCA = Maximum continuous calcium site - Superficial calcium arc (°)	
    # MCCS_MSCT = Maximum continuous calcium site - Maximum superficial calcium thickness (mm)	
    # MCCS_MINSCT = Maximum continuous calcium site - Minimum superficial calcium thickness (mm)	
    # MCCS_CSCA = Maximum continuous calcium site - Circumferential superficial calcium	
    # MCCS_CSCA_270 = Maximum continuous calcium site - Length of circumferential superficial calcium greater than equal to 270(mm)	
    # MCCS_CSCA_180 = Maximum continuous calcium site - Length of circumferential superficial calcium greater than equal to 180(mm)
    # FMSA_LA = Final minimum stent area site - Lumen area (mm2)	
    # FMSA_AS = Final minimum stent area site - Area stenosis (%)	
    # FMSA_SCA = Final minimum stent area site - Superficial calcium arc (°)	
    # FMSA_MSCT = Final minimum stent area site - Maximum superficial calcium thickness (mm)
    # ```

    selected_columns = [
        'USUBJID',
        'STUDY',
        'MLAS_AS',
        'MLAS_SCA',
        'MLAS_MSCT',
        'MCS_LA',
        'MCS_AS',
        'MCS_SCA',
        'MCS_MSCT',
        'MCCS_LA',
        'MCCS_AS',
        'MCCS_SCA',
        'MCCS_MSCT',
        'MCCS_MINSCT',
        'MCCS_CSCA',
        'MCCS_CSCA_270',
        'MCCS_CSCA_180',
        'FMSA_LA',
        'FMSA_AS',
        'FMSA_SCA',
        'FMSA_MSCT',
    ]
    if Pre_or_Post == 'Post':
        filter_out = set(('MCCS_LA', 'MCCS_AS', 'MCCS_SCA', 'MCCS_MSCT', 'MCCS_MINSCT', 'MCCS_CSCA', 'MCCS_CSCA_270', 'MCCS_CSCA_180'))
        selected_columns = [column for column in selected_columns if column not in filter_out]


    pre_or_post_ivl_df = pre_or_post_ivl_df[selected_columns]
    pre_or_post_ivl_df.head()

    # ### Just to make every easy for the dataloader, we will put the absolute path of the image to the DataFrame

    # Check that all the images are named with 'Final', 'Pre', or 'Post'

    image_path = unlabeled_data_dir / 'NiFTI'
    image_path = image_path.glob('*.nii.gz')
    for i in image_path:
        if 'Final' in i.stem or 'Pre' in i.stem or 'Post' in i.stem:
            continue
        else:
            print(i.stem)

    def resolve_image_path(subject_id: str):
        image_path = unlabeled_data_dir / 'NiFTI'
        image_path = image_path.glob("{}{}*".format(subject_id.replace('-', ''), Pre_or_Post))
        image_path = list(image_path)
        if not image_path:
            return None
        return image_path[0]

    pre_or_post_ivl_df['image_path'] = pre_or_post_ivl_df['USUBJID'].apply(lambda x: resolve_image_path(x))

    pre_or_post_ivl_df.dropna(subset=['image_path'], inplace=True)


    # ### Lastly fill . with null values
    pre_or_post_ivl_df.replace({'\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0.': np.nan}, inplace=False).to_csv(f'tabular_data/{Pre_or_Post}_IVL.csv', index=False)


import numpy as np
import re
import unidecode

post_stent = pd.read_excel('tabular_data/T6_POST_STENT_LESION.xlsx', skiprows=4)

column_names = post_stent.columns
column_names = {column_name:column_name.split(' = ')[0] for column_name in column_names}

post_stent.rename(columns=column_names, inplace=True)


def format_subject_id(subject_id: str):
    # Define the pattern
    pattern = r"[A-Z]{2} [0-9]{4,5}-([0-9]{3}-[0-9]{3})"

    # Compile the pattern
    regex = re.compile(pattern)

    # Search for the pattern in the input string
    match = regex.search(unidecode.unidecode(subject_id))

    # If a match is found, extract the desired part using the replacement pattern
    if match:
        extracted_part = re.sub(r".*?([0-9]{3}-[0-9]{3})", r"\1", match.group(1))
        return extracted_part
    raise ValueError(f"No match found {subject_id.strip().replace(u'\xa0', ' ')}")

post_stent['USUBJID'] = post_stent['USUBJID'].apply(lambda x: format_subject_id(x))

selected_columns = [
    'USUBJID',
    'STUDY',

    # Categorical
    'MAL_PRES',
    'MAL_PROX',
    'MAL_DIS',
    'MAL_SBOD',
    'MMAL_CF',
    'MMAL_NCF',
    'CF_PRES',
    'CF_3',
    'CF_2',
    'CF_1',

    # Morphological
    'MMAL_SA',
    'MMAL_LA',
    'MMAL_ARC',
    'MMAL_AR',
    'MMAL_PAR',
    'MAL_LEN',
    'MAL_THICK',
    'TOT_CAL0',
    'TOT_CAL',
    'MEAN_CF',
    'MAX_CF',
    'TOT_CFLEN',
    'MAX_CFDEP',
    'MAX_CFWID',
    'MAX_CFTHK',
    'MAX_CARC',
    'MIN_CARC',
]

post_stent = post_stent[selected_columns]

image_path = unlabeled_data_dir / 'NiFTI'
image_path = image_path.glob('*.nii.gz')
for i in image_path:
    if 'Final' in i.stem or 'Pre' in i.stem or 'Post' in i.stem:
        continue
    else:
        print(i.stem)

def resolve_image_path(subject_id: str):
    image_path = unlabeled_data_dir / 'NiFTI'
    image_path = image_path.glob("{}Final*".format(subject_id.replace('-', '')))
    image_path = list(image_path)
    if not image_path:
        return None
    return image_path[0]

post_stent['image_path'] = post_stent['USUBJID'].apply(lambda x: resolve_image_path(x))
post_stent.dropna(subset=['image_path'], inplace=True)

post_stent.replace({'\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0.': np.nan}, inplace=False).to_csv('tabular_data/Post_Stent.csv', index=False)
