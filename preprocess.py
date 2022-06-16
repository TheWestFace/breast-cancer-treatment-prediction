import argparse
import os
from glob import glob

import dicom2nifti
import dicom2nifti.settings as settings
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from p_tqdm import p_umap
from pydicom import dcmread

settings.disable_validate_slice_increment()
settings.disable_validate_slicecount()


def make_mip(image):
    """Create a Maximum Intensity Projection from an input 3D array of image slices

    Args:
        image (np.array): Input array

    Returns:
        np.array: 2D MIP
    """
    shape = np.shape(image)
    mip = np.zeros(image.shape[:2])
    for i in range(shape[0]):
        for j in range(shape[1]):
            mip[i, j] = np.max(image[i, j, :])
    return mip


def process_series(dicom_names, mips_dir, nifti_dir, root=""):
    first_file = dcmread(dicom_names[0])
    patient_id, study_id, series_id = (
        os.path.dirname(dicom_names[0]).replace(" ", "_").split("/")[-3:]
    )
    for c in ["(", ")", " ", "\\", "/", "-", ":"]:
        series_id = series_id.replace(c, "_")

    # Save all tags except the actual data
    keys = [x for x in first_file.dir() if x != "PixelData"]
    metadata = {k: str(first_file.get(k)) for k in keys}

    mip_path = os.path.join(
        mips_dir, patient_id, study_id, series_id.replace("\\ ", "_") + ".mip.png"
    )
    nifti_path = mip_path.replace(mips_dir, nifti_dir).replace(".mip.png", ".nii.gz")

    for d in [os.path.dirname(mip_path), os.path.dirname(nifti_path)]:
        os.makedirs(d, exist_ok=True)

    if len(dicom_names) > 1:
        try:
            dicom2nifti.dicom_series_to_nifti(
                os.path.dirname(dicom_names[0]),
                nifti_path,
                reorient_nifti=True,
            )
            mip = make_mip(nib.load(nifti_path).get_fdata())

            try:
                plt.imsave(mip_path, mip, cmap="gray")
            except Exception as e:
                print(f"problem writing {mip_path}: {e}")

            metadata["mip_path"] = mip_path.replace(root, "")
            metadata["nifti_path"] = nifti_path.replace(root, "")
        except Exception as e:
            print(f"Problem converting {os.path.dirname(dicom_names[0])}: {e}")
            metadata["conversion_error"] = e
    else:
        # we don't need a MIP if there is only one slice
        # also, dicom2nifti does not like only one slice
        # TODO re-run and save out a PNG of the single slice
        metadata["dicom_path"] = dicom_names[0]
    metadata["dicom_directory"] = os.path.dirname(dicom_names[0])

    return metadata


def get_dicoms(source: str):
    result = []
    if os.path.isdir(source):
        for (dirpath, _, filenames) in os.walk(source):
            dicom_names = [
                os.path.join(dirpath, x)
                for x in filenames
                if x.endswith("dcm") or x.endswith("dicom")
            ]
            if len(dicom_names) > 0:
                result += [dicom_names]
    else:
        result = []
        for expanded_source in glob.glob(source):
            result += get_dicoms(expanded_source)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_directory",
        help="input directory with dicom files to preprocess",
        type=str,
    )
    parser.add_argument(
        "nifti_directory",
        help="output directory for nifti files",
        type=str,
    )
    parser.add_argument(
        "mip_directory",
        help="output directory for mips",
        type=str,
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="overwrite series metadata file if it exists",
    )
    parser.add_argument("--num_cpus", default=2, type=int)
    parser.add_argument(
        "--root", help="specify paths relative to ROOT", type=str, default=""
    )
    args = parser.parse_args()
    dicoms = get_dicoms(args.input_directory)

    work = [
        (dicom_names, args.mip_directory, args.nifti_directory, args.root)
        for dicom_names in dicoms
    ]
    results = p_umap(
        process_series,
        dicoms,
        [args.mip_directory for d in dicoms],
        [args.nifti_directory for d in dicoms],
        [args.root for d in dicoms],
        num_cpus=args.num_cpus,
    )

    df = pd.DataFrame(results)
    df.to_csv(args.input_directory.replace("/", "_") + ".metadata.csv")
