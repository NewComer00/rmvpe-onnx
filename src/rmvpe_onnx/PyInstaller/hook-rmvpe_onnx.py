from PyInstaller.utils.hooks import collect_data_files

# Collects data/rmvpe.onnx (present after `rmvpe-onnx download`)
# and any other package data automatically.
datas = collect_data_files("rmvpe_onnx")
