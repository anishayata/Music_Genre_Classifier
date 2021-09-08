import json
import os
import librosa
import math


DATASET_PATH = "genres"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfccs(dataset_path, json_path, n_mfcc=13, n_fft= 2048, hop_length= 512, num_segments=10):

    # dictionary to hold mappings
    data = {
        "mapping": [],
        "mfcc": [],
        "label": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the files in all the genre folders in the dataset
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            # add semantic labels
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

        #  process files in the current genre folder
        for f in filenames:

            # load audio files
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # process segments, extracting mfccs and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)

                mfcc = mfcc.T

                # store mfcc for segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["label"].append(i-1)
                    print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfccs(DATASET_PATH, JSON_PATH, num_segments=10)