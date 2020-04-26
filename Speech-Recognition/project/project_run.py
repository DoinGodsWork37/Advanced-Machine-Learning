import prep_data
import model
import os
from pathlib import Path
from sklearn.externals import joblib

os.chdir(os.path.join(os.getcwd(), "Speech-Recognition/project"))

paths = Path(r"sample").glob("**/*.wav")
splits = list(map(prep_data.split_numbers, paths))
splits_dict = {k: [d[k] for d in splits] for k in splits[0]}

prep_data.save_splits(splits_dict, "new_data")

augmented_data = list(map(prep_data.augment,
                          ["new_data/0", "new_data/1", "new_data/2",
                           "new_data/3","new_data/4", "new_data/5",
                           "new_data/6", "new_data/7", "new_data/8",
                           "new_data/9"]))


augmented_dict = {k: v for item in augmented_data for k, v in item.items()}

prep_data.save_augments(augmented_dict, "new_data")

num_models = model.build_models("new_data/")

joblib.dump(num_models, "saved_num_models.pkl")