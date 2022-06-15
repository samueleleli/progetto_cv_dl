import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

tipo="arch9l"


print("TIPO DATASET:",tipo)
if tipo=="s3dis":
    import indoor3d_util
    data_name = "Stanford3dDataset_v1.2_Aligned_Version"
    DATA_PATH = os.path.join(ROOT_DIR, 'data', data_name)
    META_PATH = os.path.join(BASE_DIR, "meta")
    anno_paths = [line.rstrip() for line in open(os.path.join(META_PATH, 'anno_paths.txt'))]
    output_folder = os.path.join(DATA_PATH, 'stanford_indoor3d')

elif tipo=="arch":
    import indoor3d_util_arch as indoor3d_util
    data_name = "arch"
    DATA_PATH = os.path.join(ROOT_DIR, 'data', data_name)
    META_PATH = os.path.join(DATA_PATH, "meta")
    anno_paths = [line.rstrip() for line in open(os.path.join(META_PATH, 'anno_paths.txt'))]
    output_folder = os.path.join(ROOT_DIR, 'data', 'arch-npy')

elif tipo=="arch9l":
    import indoor3d_util_arch9l as indoor3d_util
    data_name = "arch9l"
    DATA_PATH = os.path.join(ROOT_DIR, 'data', data_name)
    META_PATH = os.path.join(DATA_PATH, "meta")
    anno_paths = [line.rstrip() for line in open(os.path.join(META_PATH, 'anno_paths.txt'))]
    output_folder = os.path.join(ROOT_DIR, 'data', 'arch9l-npy')

elif tipo=="sinthcity":
    import indoor3d_util_sinthcity as indoor3d_util
    data_name = "sinthcity"
    DATA_PATH = os.path.join(ROOT_DIR, 'data', data_name)
    META_PATH = os.path.join(DATA_PATH, "meta")
    anno_paths = [line.rstrip() for line in open(os.path.join(META_PATH, 'anno_paths.txt'))]
    output_folder = os.path.join(ROOT_DIR, 'data', data_name+'-npy')


anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    output_folder = os.path.join(output_folder, data_name)
    os.mkdir(output_folder)    #stesso nome finale di DATA_PATH

npy_files = []
for anno_path in anno_paths:
    print(anno_path)
    anno_path = anno_path.replace("\\","/")
    elements = anno_path.split('/')
    out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
    npy_files.append(out_filename)
    indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')

'''
occorre creare in meta/ il file all_data_label.txt contenente i nomi dei .npy creati
'''
all_data_label_path = os.path.join(META_PATH, "all_data_label.txt")
with open(all_data_label_path, "w") as fw:
    for f in npy_files:
        fw.write("{}\n".format(f))

print("Done!")