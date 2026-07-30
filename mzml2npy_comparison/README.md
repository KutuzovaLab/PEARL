How to run:

Add "/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/matrix_standards/Negative/for_yucca/test_dir/images" as "input/gold_standard/images/"
Add "/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/matrix_standards/Negative/for_yucca/test_dir/labels" as "input/gold_standard/labels/"

Add “/projects/zcr545_shared_datasets/data/projects/DL_LC/DSP_backup/matrix_standards/Negative/mzml_files” as "input/mzml"
Also copy "/home/frz603/DSP/matrix_standards/Standards_NEG.xlsx" into "input/mzml"

then run (full run, about 16 hours)

python main.py

or for only 1 file

python main.py --limit 1

or for only optimized matched against gold standard

python main.py --skip-original