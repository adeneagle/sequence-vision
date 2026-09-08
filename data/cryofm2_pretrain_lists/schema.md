# CryoFM2 Dataset Schema

This document describes the schema for all datasets in the CryoFM2 collection.

## Dataset Overview

CryoFM2 contains three datasets with different schemas depending on the task:

1. **cryofm2_pretrain_dataset**: Pre-training dataset with half-map pairs
2. **cryofm2_emhancer_dataset**: Enhancement dataset with half-map pairs and model-based maps
3. **cryofm2_emready_dataset**: EMReady dataset with deposited and simulated maps

## cryofm2_pretrain_dataset

### Schema

| Field Name | Type | Description |
|------------|------|-------------|
| `map_path1` | string | Relative path to the first half-map |
| `map_path2` | string | Relative path to the second half-map |
| `mean1` | float | Mean value of the first map |
| `std1` | float | Standard deviation of the first map |
| `quantile_max_value1` | float | Quantile-based maximum value of the first map |
| `mean2` | float | Mean value of the second map |
| `std2` | float | Standard deviation of the second map |
| `quantile_max_value2` | float | Quantile-based maximum value of the second map |
| `apix` | float | Pixel size in Angstroms per pixel |
| `emdb_id` | string | EMDB entry ID (e.g., "EMD-12042") |

### Files
- `train.csv`: Training set
- `test.csv`: Test set

### Example Entry
```csv
map_path1,map_path2,mean1,std1,quantile_max_value1,mean2,std2,quantile_max_value2,apix,emdb_id
EMD-12042/other/emd_12042_half_map_1.mrc,EMD-12042/other/emd_12042_half_map_2.mrc,7.1475265e-06,0.003149332,0.10584783387556276,6.5108984e-06,0.0031298543,0.1064059325212141,1.4967,EMD-12042
```

### Note on Data Format
During pre-training, only the map corresponding to `map_path2` is used for each row. To facilitate usage, each half-map pair appears twice in the CSV: once with `map_path1` and `map_path2` in their original order, and once with the paths swapped. This design allows each half-map to be used as the target (`map_path2`) in the training process.

## cryofm2_emhancer_dataset

### Schema

| Field Name | Type | Description |
|------------|------|-------------|
| `map_path1` | string | Relative path to the first map (half-map) |
| `map_path2` | string | Relative path to the second map (model-based LocScale map) |
| `mean1` | float | Mean value of the first map |
| `std1` | float | Standard deviation of the first map |
| `quantile_max_value1` | float | Quantile-based maximum value of the first map |
| `mean2` | float | Mean value of the second map |
| `std2` | float | Standard deviation of the second map |
| `quantile_max_value2` | float | Quantile-based maximum value of the second map |
| `apix` | float | Pixel size in Angstroms per pixel |
| `emdb_id` | string | EMDB entry ID (e.g., "EMD-0026") |

### Files
- `train.csv`: Training set
- `validation.csv`: Validation set
- `test_info.csv`: Test set metadata (see below)

### Example Entry (train.csv / validation.csv)
```csv
map_path1,map_path2,mean1,std1,quantile_max_value1,mean2,std2,quantile_max_value2,apix,emdb_id
EMD-0026/half_map_1.mrc,EMD-0026/model_based_locscale.mrc,7.066989e-06,0.0011755731,0.026576576789403,0.0005114955,0.008879976,0.3861291066059188,1.5,EMD-0026
```

### test_info.csv Schema

| Field Name | Type | Description |
|------------|------|-------------|
| `pdb_id` | string | PDB entry ID (e.g., "5vkq") |
| `emdb_id` | string | EMDB entry ID (e.g., "8702") |
| `resolution` | float | Resolution in Angstroms |

### Example Entry (test_info.csv)
```csv
pdb_id, emdb_id, resolution
5vkq, 8702, 3.55
```

## cryofm2_emready_dataset

### Schema

| Field Name | Type | Description |
|------------|------|-------------|
| `map_path1` | string | Relative path to the deposited map |
| `map_path2` | string | Relative path to the simulated map |
| `mean1` | float | Mean value of the deposited map |
| `std1` | float | Standard deviation of the deposited map |
| `quantile_max_value1` | float | Quantile-based maximum value of the deposited map |
| `apix` | float | Pixel size in Angstroms per pixel |
| `pdb_id` | string | PDB entry ID (e.g., "6GYB") |
| `emdb_id` | string | EMDB entry ID (e.g., "0089") |

### Files
- `train.csv`: Training set
- `val.csv`: Validation set
- `test_info.csv`: Test set metadata (same schema as emhancer dataset)

### Example Entry (train.csv / val.csv)
```csv
map_path1,map_path2,mean1,std1,quantile_max_value1,apix,pdb_id,emdb_id
6GYB/deposited.mrc,6GYB/simulated.mrc,0.0042865635,0.015467119,0.1621950377963379,1.5125,6GYB,0089
```

### Note on Data Format
The `map_path2` corresponds to a simulated map with density values in the range [0, 1]. As a result, no per-map normalization based on the map's own statistics is required for `map_path2`.

### test_info.csv Schema

Same as `cryofm2_emhancer_dataset/test_info.csv`:
- `pdb_id`: PDB entry ID
- `emdb_id`: EMDB entry ID
- `resolution`: Resolution in Angstroms

## Field Descriptions

### Map Paths
- **`map_path1`, `map_path2`**: Relative paths to map files in MRC format
- Paths are relative to the dataset root directory
- Actual map files should be downloaded from EMDB or provided separately

### Statistical Features
- **`mean1`, `mean2`**: Mean intensity values of the maps
- **`std1`, `std2`**: Standard deviation of intensity values
- **`quantile_max_value1`, `quantile_max_value2`**: Quantile-based maximum values (used for normalization)

### Resolution
- **`apix`**: Pixel size in Angstroms per pixel (lower values = higher resolution)
- **`resolution`** (in test_info.csv): Overall resolution of the structure in Angstroms

### Identifiers
- **`emdb_id`**: Electron Microscopy Data Bank entry ID
- **`pdb_id`**: Protein Data Bank entry ID (when available)

## Notes

- All map files are in MRC format (`.mrc` extension)
- Half-maps and deposited maps are downloaded from EMDB and resized to 1.5 $\AA$/pixel
- Model-based LocScale maps are generated according to the method described in the DeepEMhancer paper 
- Simulated maps are provided by the EMReady authors
- The actual map files are not included in this repository; only the metadata lists are provided

