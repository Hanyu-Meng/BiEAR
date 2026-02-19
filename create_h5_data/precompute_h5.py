from data_h5_save import save_dataset_to_h5

ROOT = "/media/mengh/SharedData/hanyu/DeepEar/spirit_dataset"

# save_dataset_to_h5(
#     dataset_dir=f"{ROOT}/anechoic_train",
#     h5_path=f"{ROOT}/anechoic_train_active_wav.h5",
#     max_samples=1000,
#     compression=None,
# )

# save_dataset_to_h5(
#     dataset_dir=f"{ROOT}/anechoic_val",
#     h5_path=f"{ROOT}/anechoic_val_active_wav.h5",
#     max_samples=1000,
#     compression=None,
# )

# save_dataset_to_h5(dataset_dir=f"{ROOT}/spirit_test",
#                    h5_path=f"{ROOT}/spirit_test1_active_wav.h5",
#                    max_samples=None,
#                    compression=None)
# all from unseen speakers
save_dataset_to_h5(dataset_dir=f"{ROOT}/spirit_test2",
                   h5_path=f"{ROOT}/spirit_test2_active_wav.h5",
                   max_samples=None,
                   compression=None)