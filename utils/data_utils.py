# Copyright 2020 - 2022 MONAI Consortium
# Modified by Jiatian Zhang, 2025, for BDFM
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *
import torch
from monai.config import DtypeLike, KeysCollection
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class changToImage(MapTransform):
    def __call__(self, data):
        d = dict(data)
        d["image"] = d[self.keys[0]]
        del d[self.keys[0]]

        return d


class SafeScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    Removes image from output if max and min values are the same to avoid producing NaNs.
    """

    backend = ScaleIntensity.backend

    def __init__(
            self,
            keys: KeysCollection,
            minv: Optional[float] = 0.0,
            maxv: Optional[float] = 1.0,
            factor: Optional[float] = None,
            channel_wise: bool = False,
            dtype: DtypeLike = np.float32,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensity(minv, maxv, factor, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img is not None:
                # Check if the max and min are the same
                if np.max(img) == np.min(img):
                    # If true, skip adding this image to the result (effectively removing it)
                    continue
                else:
                    # Proceed with normal scaling
                    d[key] = self.scaler(img)
            else:
                # Optionally handle None images here if needed
                pass
        return d


def apply_transform_safe(transform, data):
    try:
        return transform(data)
    except Exception as e:
        logging.error(f"Error applying transform: {e}")
        logging.error(f"Data causing the issue: {data}")
        raise


class TrackingDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform=transform)

    def __getitem__(self, index):
        try:
            data = self.data[index]
            if self.transform:
                data = apply_transform_safe(self.transform, data)
            return data
        except Exception as e:
            logging.error(f"Error processing index {index}: {e}")
            raise


class SafeLoadImaged(LoadImaged):
    def __call__(self, data):
        try:
            return super().__call__(data)
        except Exception as e:
            print(f"Skipping corrupt data: {data['image']} due to error: {e}")
            return None


class PrintShapeTransform:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        image = data[self.keys[0]]
        print(f"Current shape of {self.keys[0]}: {image.shape}")
        return data


def get_loader(args):
    splits1 = "/BraTS23_GLI_onemodal.json"
    splits2 = "/BraTS23_MEN_onemodal.json"
    splits3 = "/BraTS23_MET_onemodal.json"
    splits4 = "/BraTS23_PED_onemodal.json"
    splits5 = "/BraTS23_SSA_onemodal.json"

    splits6 = "/AtlasR2.json"
    splits7 = "/BrainPTM2021.json"
    splits8 = "/ISLES2022.json"
    splits9 = "/OASIS.json"
    splits10 = "/UPENN-GBM"
    splits11 = "/MRBrains13"

    list_dir = "./jsons/"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7
    jsonlist8 = list_dir + splits8
    jsonlist9 = list_dir + splits9
    jsonlist10 = list_dir + splits10
    jsonlist11 = list_dir + splits11

    datadir1 = "./datasets/BraTS23_GLI"
    datadir2 = "./datasets/BraTS23_MEN"
    datadir3 = "./datasets/BraTS23_MET"
    datadir4 = "./datasets/BraTS23_PED"
    datadir5 = "./datasets/AtlasR2"
    datadir6 = "./datasets/FUDAN/7-Cerebral_aneurysm"
    datadir7 = "./datasets/BrainPTM2021"
    datadir8 = "./datasets/ISLES2022"
    datadir9 = "./datasets/OASIS"
    datadir10 = "./datasets/ISLES2022"
    datadir11 = "./datasets/OASIS"

    num_workers = 0

    # brats23-gli
    datalist1_train = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    datalist1_val = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    datalist1 = datalist1_train + datalist1_val
    print("Dataset 1 GLI: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    new_datalist1_val = []

    for item in datalist1_train:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    for item in datalist1_val:
        item_dict = {"image": item["image"]}
        new_datalist1_val.append(item_dict)

    # brats23-men
    datalist2_train = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    datalist2_val = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    datalist2 = datalist2_train + datalist2_val
    print("Dataset 2 MEN: number of data: {}".format(len(datalist2)))
    new_datalist2 = []
    new_datalist2_val = []
    for item in datalist2_train:
        item_dict = {"image": item["image"]}
        new_datalist2.append(item_dict)
    for item in new_datalist2_val:
        item_dict = {"image": item["image"]}
        new_datalist2_val.append(item_dict)

    # brats23-met
    datalist3_train = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    datalist3_val = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    datalist3 = datalist3_train + datalist3_val
    print("Dataset 3 MET: number of data: {}".format(len(datalist3)))
    new_datalist3 = []
    new_datalist3_val = []
    for item in datalist3_train:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)
    for item in datalist3_val:
        item_dict = {"image": item["image"]}
        new_datalist3_val.append(item_dict)

    # brats23-ped
    datalist4_train = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    datalist4_val = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
    datalist4 = datalist4_train + datalist4_val
    print("Dataset 4 PED: number of data: {}".format(len(datalist4)))
    new_datalist4 = []
    new_datalist4_val = []
    for item in datalist4_train:
        item_dict = {"image": item["image"]}
        new_datalist4.append(item_dict)
    for item in datalist4_val:
        item_dict = {"image": item["image"]}
        new_datalist4_val.append(item_dict)

    # brats23-ssa
    datalist5_train = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    datalist5_val = load_decathlon_datalist(jsonlist5, False, "validation", base_dir=datadir5)
    datalist5 = datalist5_train + datalist5_val
    print("Dataset 5 SSA: number of data: {}".format(len(datalist5)))
    new_datalist5 = []
    new_datalist5_val = []
    for item in datalist5_train:
        item_dict = {"image": item["image"]}
        new_datalist5.append(item_dict)
    for item in datalist5_val:
        item_dict = {"image": item["image"]}
        new_datalist5_val.append(item_dict)

    # atlasR2
    datalist6_train = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    datalist6_val = load_decathlon_datalist(jsonlist6, False, "validation", base_dir=datadir6)
    datalist6 = datalist6_train + datalist6_val
    print("Dataset 6 AtlasR2: number of data: {}".format(len(datalist6)))
    new_datalist6 = []
    new_datalist6_val = []
    for item in datalist6_train:
        item_dict = {"image": item["image"]}
        new_datalist6.append(item_dict)
    for item in datalist6_val:
        item_dict = {"image": item["image"]}
        new_datalist6_val.append(item_dict)

    # BrainPTM2021
    datalist7_train = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    datalist7_val = load_decathlon_datalist(jsonlist7, False, "validation", base_dir=datadir7)
    datalist7 = datalist7_train + datalist7_val
    print("Dataset 7 BrainPTM2021: number of data: {}".format(len(datalist7)))
    new_datalist7 = []
    new_datalist7_val = []
    for item in datalist7_train:
        item_dict = {"image": item["image"]}
        new_datalist7.append(item_dict)
    for item in datalist7_val:
        item_dict = {"image": item["image"]}
        new_datalist7_val.append(item_dict)

    # ISLES2022
    datalist8_train = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
    datalist8_val = load_decathlon_datalist(jsonlist8, False, "validation", base_dir=datadir8)
    datalist8 = datalist8_train + datalist8_val
    print("Dataset 8 ISLES2022: number of data: {}".format(len(datalist8)))
    new_datalist8 = []
    new_datalist8_val = []
    for item in datalist8_train:
        item_dict = {"image": item["image"]}
        new_datalist8.append(item_dict)
    for item in datalist8_val:
        item_dict = {"image": item["image"]}
        new_datalist8_val.append(item_dict)

    # oasis
    datalist9_train = load_decathlon_datalist(jsonlist9, False, "training", base_dir=datadir9)
    datalist9_val = load_decathlon_datalist(jsonlist9, False, "validation", base_dir=datadir9)
    datalist9 = datalist9_train + datalist9_val
    print("Dataset 9 OASIS: number of data: {}".format(len(datalist9)))
    new_datalist9 = []
    new_datalist9_val = []
    for item in datalist9_train:
        item_dict = {"image": item["image"]}
        new_datalist9.append(item_dict)
    for item in datalist9_val:
        item_dict = {"image": item["image"]}
        new_datalist9_val.append(item_dict)

    # upenn-gbm
    datalist10_train = load_decathlon_datalist(jsonlist10, False, "training", base_dir=datadir10)
    datalist10_val = load_decathlon_datalist(jsonlist10, False, "validation", base_dir=datadir10)
    datalist10 = datalist10_train + datalist10_val
    print("Dataset 10 UPENN-GBM: number of data: {}".format(len(datalist10)))
    new_datalist10 = []
    new_datalist10_val = []
    for item in datalist10_train:
        item_dict = {"image": item["image"]}
        new_datalist10.append(item_dict)
    for item in datalist10_val:
        item_dict = {"image": item["image"]}
        new_datalist10_val.append(item_dict)

    # MRBrains13
    datalist11_train = load_decathlon_datalist(jsonlist11, False, "training", base_dir=datadir11)
    datalist11_val = load_decathlon_datalist(jsonlist11, False, "validation", base_dir=datadir11)
    datalist11 = datalist11_train + datalist11_val
    print("Dataset 11 MRBrains13: number of data: {}".format(len(datalist11)))
    new_datalist11 = []
    new_datalist11_val = []
    for item in datalist11_train:
        item_dict = {"image": item["image"]}
        new_datalist11.append(item_dict)
    for item in datalist11_val:
        item_dict = {"image": item["image"]}
        new_datalist11_val.append(item_dict)

    datalist = new_datalist1 + new_datalist2 + new_datalist3 + new_datalist4 + new_datalist5 \
               + new_datalist6 + new_datalist7 + new_datalist8 + new_datalist9 + new_datalist10 + new_datalist11
    datalist_val = new_datalist1_val + new_datalist2_val + new_datalist3_val + new_datalist4_val \
                   + new_datalist5_val + new_datalist6_val + new_datalist7_val + new_datalist8_val \
                   + new_datalist9_val + new_datalist10_val + new_datalist11_val

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(datalist_val)))

    train_transforms = Compose([
        SafeLoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        # PrintShapeTransform(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        # PercentileClipIntensityd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        SafeScaleIntensityd(keys=["image"], channel_wise=True, minv=0., maxv=1.),
        # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=(96, 96, 96),
            random_size=False,
            num_samples=1,
        ),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), mode='constant', constant_values=0),
        RandFlipd(keys="image", prob=0.1, spatial_axis=0),
        RandFlipd(keys="image", prob=0.1, spatial_axis=1),
        RandFlipd(keys="image", prob=0.1, spatial_axis=2),
        RandRotate90d(keys="image", prob=0.1, max_k=3),
        ToTensord(keys=["image"])
    ])

    val_transforms = Compose([
        SafeLoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        # PercentileClipIntensityd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        SafeScaleIntensityd(keys=["image"], channel_wise=True, minv=0., maxv=1.),
        # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=(96, 96, 96),
            random_size=False,
            num_samples=1,
        ),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96), mode='constant', constant_values=0),
        ToTensord(keys=["image"])
    ])

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms,
                                cache_rate=0.5, num_workers=num_workers)
        val_ds = CacheDataset(data=datalist_val, transform=val_transforms,
                              cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset")
        train_ds = TrackingDataset(data=datalist, transform=train_transforms)
        val_ds = TrackingDataset(data=datalist_val, transform=val_transforms)
    print('over')

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=True,
        drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=num_workers, sampler=val_sampler, shuffle=True,
        drop_last=True, pin_memory=True
    )

    loader = [train_loader, val_loader]
    return loader