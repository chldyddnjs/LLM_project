import zipfile
import os
import csv
import json
import multiprocessing as mp
import argparse
import logging
from PIL import Image
import os

import datasets
import json
from itertools import chain

_CITATION = """\
    논문 등의 인용 서지 정보
"""

_DESCRIPTION = """\
    데이콘 이미지-텍스트 멀티모달 데이터
"""

_HOMEPAGE = "URL"

_LICENSE = "Dacon"

path = "/root/Data_hub/dacon"

def instruct_format_func(data):
    
    id, image_id, question, answer = data
    return {
        "id": id,
        "image": Image.open(os.path.join(path,'image/train',image_id + ".jpg")),
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }

class DaconConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class DaconDataset(datasets.GeneratorBasedBuilder):


    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "image": datasets.Image(),
                "conversations": datasets.Sequence(
                    datasets.Features(
                        {
                            "from": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }),
                )
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # download_files = dl_manager.download_and_extract(_URL)
        # data_dir = os.path.dirname(download_files)
        
        train_path = os.path.join(path,'train.csv')
        test_path = os.path.join(path,'test.csv')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, f"{task_name}.jsonl"),
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath":test_path,
            #     },
            # ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath,"r",encoding="utf-8") as f:
            reader = csv.reader(f)
            for key, row in enumerate(reader):
                if key == 0: continue
                yield key, instruct_format_func(row)