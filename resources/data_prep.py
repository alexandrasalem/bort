import argparse
import os
import random
import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


from bort.resources.logs import logger
from bort.resources.datasets_talkbank import ErrorSessionCollection
from bort.resources.yaml_to_fairseq_data import all_source_target_texts


RANDOM_SEED = 8675309
random.seed(RANDOM_SEED)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--parallel", type=int, default=8)
    return parser.parse_args()


def main(out_dir: str, parallel: int):
    yaml_file = os.path.join(out_dir, "prepared-data.yaml")

    fairseq_data_dir = os.path.join(out_dir, "fairseq")
    sessions = ErrorSessionCollection.from_yaml(yaml_file)

    prepare_fairseq_files(sessions, out_dir=fairseq_data_dir)

    fairseq_data_dir = os.path.join(out_dir, "fairseq_target_prons")
    prepare_fairseq_files(sessions, out_dir=fairseq_data_dir, use_target_pron=True)

    logger.info("Done!")


def prepare_fairseq_files(sessions, *, out_dir, use_target_pron=False):
    out = {}
    out["participant_ids"], out["session_ids"], out["gem_names"], out["source"], out["target"], out["missing_pronunciations"], out["chat_production"], out["error_type"] = all_source_target_texts(
        sessions=sessions,
        use_target_pron=use_target_pron
    )

    simple_error_type = []
    for error in out["error_type"]:
        if error == "[*]":
            simple_error_type.append("unknown")
        elif error == "multiple":
            simple_error_type.append("multiple")
        else:
            simple_error_type.append(error[3])
    out["simple_error_type"] = simple_error_type

    # writing all data
    for out_key, out_data in out.items():
        out_filename = os.path.join(out_dir, f"{out_key}.txt")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w") as f:
            f.write("\n".join(out_data))

    # writing splits
    splitter = GroupKFold(n_splits=10)
    splits = splitter.split(out["source"], out["target"], groups=out["participant_ids"])
    sets = []
    for _, test in splits:
        sets.append(list(test))
    for i in tqdm(range(10), desc="Building splits"):
        sets_copy = sets.copy()
        if i == 9:
            valid = sets_copy.pop(i)
            test = sets_copy.pop(0)
        else:
            valid = sets_copy.pop(i)
            test = sets_copy.pop(i)
        train = sum(sets_copy, [])

        #checking there's no overlap:
        for idx in train:
            if idx in test:
                raise ValueError
            elif idx in valid:
                raise ValueError
        for idx in valid:
            if idx in train:
                raise ValueError
            elif idx in test:
                raise ValueError

        test = np.array(test)
        valid = np.array(valid)
        train = np.array(train)

        for out_key, out_data in out.items():
            if out_key == "missing_pronunciations":
                continue
            out_data_train = list(np.array(out_data)[train])
            out_data_valid = list(np.array(out_data)[valid])
            out_data_test = list(np.array(out_data)[test])

            out_filename_train = os.path.join(out_dir, f"folds/fold_{i}/splits/train/{out_key}.txt")
            os.makedirs(os.path.dirname(out_filename_train), exist_ok=True)
            with open(out_filename_train, "w") as f:
                f.write("\n".join(out_data_train))

            out_filename_valid = os.path.join(out_dir, f"folds/fold_{i}/splits/valid/{out_key}.txt")
            os.makedirs(os.path.dirname(out_filename_valid), exist_ok=True)
            with open(out_filename_valid, "w") as f:
                f.write("\n".join(out_data_valid))

            out_filename_test = os.path.join(out_dir, f"folds/fold_{i}/splits/test/{out_key}.txt")
            os.makedirs(os.path.dirname(out_filename_test), exist_ok=True)
            with open(out_filename_test, "w") as f:
                f.write("\n".join(out_data_test))


if __name__ == '__main__':
    args = get_args()
    main(**vars(args))