from monai.data import Dataset
from typing import Callable
import os
from tqdm import tqdm

def build_dataset_file_list(
    root_dir: str,
    file_names: list[str],
    sbj_include: str | None = None,
    sbj_exclude: str | None = None,
    n_subjects: int | None = None,
) -> list[dict]:
    """
    Build a MONAI-compatible list of file dictionaries from a structured dataset directory.

    Args:
        root_dir (str): Root directory containing subject subfolders.
        file_names (list[str]): List of filenames to look for within each subject folder.
                                Example: ["inp.nii.gz", "wat.nii.gz"]
        sbj_include (str | None): Optional path to a file containing specific subject IDs to include.
        sbj_exclude (str | None): Optional path to a file containing specific subject IDs to exclude.
        n_subjects (int | None): Optional maximum number of subjects to load (for debugging / limiting dataset).

    Returns:
        list[dict]: A list of dictionaries, each like {"image": <file_path>}.

    Example:
        >>> train_files = build_dataset_file_list(
        ...     root_dir="/data/whole_body/nifti",
        ...     file_names=["inp.nii.gz"],
        ...     sbj_include="include_subjects.txt"
        ... )
        >>> train_files[0]
        {'image': '/data/whole_body/nifti/2033420/inp.nii.gz'}
    """
    all_files = []
    if sbj_exclude is not None and os.path.isfile(sbj_exclude):
        with open(sbj_exclude, "r") as f:
            subjects_exclude = [line.strip() for line in f if line.strip()]

    # --- Load subject IDs ---
    if sbj_include is not None and os.path.isfile(sbj_include):
        with open(sbj_include, "r") as f:
            subjects = [line.strip() for line in f if line.strip()]
    else:
        subjects = sorted(os.listdir(root_dir))

    if sbj_exclude is not None and os.path.isfile(sbj_exclude):
        subjects = [sub for sub in subjects if sub not in subjects_exclude]

    # --- Iterate through subjects ---
    for subject in tqdm(subjects, desc="Building file list"):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        # --- Limit number of subjects ---
        if n_subjects is not None and len(all_files) >= n_subjects:
            break

        # --- Collect available files for each subject ---
        for file_name in file_names:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.isfile(file_path):
                all_files.append({
                    "image": file_path,
                    "sbj": subject,
                    "file_name": os.path.splitext(os.path.basename(file_name))[0],
                })

    all_files = sorted(all_files, key=lambda x: x["image"])
    assert len(all_files) > 0, f"No files found for {root_dir}!"

    print(f"Found {len(all_files)} files in {root_dir}")
    print(all_files[:10])
    return all_files

class MonaiDataset(Dataset):
    def __init__(self, data_path: str, 
        file_names: list[str], 
        sbj_include: str | None = None, 
        sbj_exclude: str | None = None, 
        n_subjects: int | None = None, 
        split: list[float] = [0.8, 0.1, 0.1],
        transforms: list[Callable] = None,
        mode: str = "train",
        **kwargs
    ) -> None:
        assert mode in ["train", "eval", "test"], "Mode must be train, eval, or test"
        assert sum(split) == 1.0, "Split must sum to 1.0"
        assert len(split) == 3, "Split must have 3 elements (train, eval, test)"
        self.data_path = data_path
        self.file_names = file_names
        self.sbj_include = sbj_include
        self.sbj_exclude = sbj_exclude
        self.n_subjects = n_subjects
        self.mode = mode
        self.split = split

        if self.mode == "train":
            self.dataset = build_dataset_file_list(self.data_path, self.file_names, self.sbj_include, self.sbj_exclude, self.n_subjects)
        elif self.mode == "eval":
            self.dataset = build_dataset_file_list(self.data_path, self.file_names, self.sbj_include, self.sbj_exclude, self.n_subjects)
        elif self.mode == "test":
            self.dataset = build_dataset_file_list(self.data_path, self.file_names, self.sbj_include, self.sbj_exclude, self.n_subjects)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        self.transforms = transforms
        super().__init__(self.dataset, self.transforms)